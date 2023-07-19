import haiku as hk
import fnx, threading, jax.tree_util, inspect, contextlib, functools
from functools import partial
import haiku._src.base as hk_base
import jax.numpy as jnp
import numpy as np

_thread_local = threading.local()
def _reapers():
    if not hasattr(_thread_local, "reapers"):
        _thread_local.reapers = []
    return _thread_local.reapers

SELF_NAME = "~"

class Reaper:
    """A context manager that reaps values sowed via ``fnx.sow``.

    Example usage:

    ..  code-block:: python

        def model(x):
            fnx.sow("my_sowed_value", x) # Sows the input with name my_sowed_value
            return x + 1

        with fnx.Reaper("my_sowed_value") as reaper: # Reaps values sowed under my_sowed_value
            x = model(x)
        y = reaper.get() # Returns value previously sowed to my_sowed_value

    Parameters:
        reap: PyTree of names that should be reaped, or function ``fn: name, value -> bool``.
        prefix: Prefix for all names in ``reap``. Defaults to ``hk.experimental.current_name()``.
    """

    def __init__(self, reap, prefix=None):
        if callable(reap):
            self.filter = reap
        else:
            leaves = jax.tree_util.tree_flatten(reap)[0]
            self.filter = lambda name, value: name in leaves
        self.reap = reap
        self.prefix = (prefix if not prefix is None else hk.experimental.current_name()) + "/"
        self.sowed = {}

    def __enter__(self):
        _reapers().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _reapers().pop()

    def add(self, name, value):
        if name.startswith(self.prefix):
            name = name[len(self.prefix):]
            if name in self.sowed:
                raise ValueError(f"Has already sowed {name} into reaper")
            if self.filter(name, value):
                self.sowed[name] = value
        else:
            assert name + "/" == self.prefix or self.prefix == "~/"

    def get(self):
        """Returns the values reaped by this reaper.

        * If ``reap`` is a PyTree of strings, returns a PyTree of the same structure with leaves replaced by reaped values.
        * If ``reap`` is a function ``fn: name, value -> bool``, returns a dictionary ``{name: value}`` with all sowed values for which ``reap`` returns True.

        Returns:
            The reaped values.
        """
        if callable(self.reap):
            return {**self.sowed}
        else:
            def get(name):
                if not name in self.sowed:
                    raise ValueError(f"{name} was not sowed into reaper")
                return self.sowed[name]
            return jax.tree_util.tree_map(get, self.reap)

class ModuleCallContext:
    """Call context of a module.

    Parameters:
        fullname: Full name of the module in the calling scope.
    """

    def __init__(self, fullname):
        self.fullname = fullname

def wrap_module(func, name: str = None):
    """Can be used to convert Haiku modules to fnx modules.

    Takes as input a function that returns

    * a ``functools.partial`` object with a Haiku module and the arguments to its ``__call__`` method,

      ..  code-block:: python

          @fnx.wrap_module
          def conv(x, kernel_size, name, ...):
              return functools.partial(hk.ConvND(kernel_size, name=name, ...), x)

    * or a Haiku module, in which case the first input argument is passed to its ``__call__`` method.

      ..  code-block:: python

          @fnx.wrap_module
          def conv(x, kernel_size, name, ...):
              return hk.ConvND(kernel_size, name=name, ...)
              # module is implicitly called with input ``x``

    The function ``func`` must contain a ``name`` argument that is passed to the Haiku module. The value is determined as follows:
    
    1. The ``name`` argument passed to the resulting fnx module if it is given.
    2. The ``name`` argument passed to this function if it is not None.
    3. The default value for the ``name`` parameter of ``func`` if it exists.
    4. ``func.__name__``

    Parameters:
        func: The function that returns the created Haiku module.
        name: The default name for the wrapped module, or None.

    Returns:
        The wrapped function.
    """

    if not name is None:
        default_name = name
    else:
        params = dict(inspect.signature(func).parameters.items())
        if not "name" in params or params["name"].default is inspect.Parameter.empty:
            default_name = func.__name__
        else:
            default_name = params["name"].default

    @functools.wraps(func)
    def module_wrapper(*args, return_module=False, reap=None, name=default_name, **kwargs):
        if "skip_module" in kwargs:
            if kwargs["skip_module"] != True:
                raise ValueError("skip_module must be True if given")
            return func(*args, name=name, **kwargs)

        if reap is None:
            reap = SELF_NAME

        output = func(*args, name=name, **kwargs)
        if isinstance(output, hk.Module):
            module = output
            module_func = partial(module, args[0])
        elif isinstance(output, partial):
            module = output.func
            if not isinstance(module, hk.Module):
                raise ValueError("func argument of functools.partial must be instance of hk.Module")
            module_func = output
        else:
            raise ValueError(f"Invalid return value with type {type(output)} of functional module {func}")

        with Reaper(reap, prefix=module.module_name) as reaper:
            output = module_func()
            sow(SELF_NAME, output, prefix=module.module_name)

        reap = reaper.get()
        if return_module:
            return reap, module
        else:
            return reap

    def context(*args, name=default_name, **kwargs):
        names = hk.experimental.current_name().split("/") + [name]
        return ModuleCallContext(fullname="/".join(names))

    from .intercept import wrap
    return wrap(module_wrapper, context)

def module(func = None, name: str = None, is_unbound_method: bool = False):
    """Creates a fnx module from the given function.

    The result is a callable with the following additional functionality:

    1. Adds a ``name`` argument for the module (see below).
    2. Adds a ``reap`` argument that determines which values should be returned from the function. See :class:`fnx.Reaper`.
    3. Allows out-of-place modification via ``fnx.intercept.*``.

    If ``func`` is ``None``, returns a partial application of ``fnx.module`` with the remaining arguments. Can be used to define arguments when used as a function decorator:

    ..  code-block:: python

        @fnx.module(name="my_module_name")
        def my_module(x):
            ...

    The (relative) name of the resulting fnx module is determined as follows:

    1. The ``name`` argument passed to the fnx module if it is given.
    2. The ``name`` argument passed to this function if it is not None.
    3. ``func.__name__``

    Parameters:
        func: The function that is converted into a fnx module.
        name: Defines the default name of the module. Defaults to ``func.__name__``.
        is_unbound_method: Should be set to ``True`` if ``func`` is an unbound method. Defaults to ``False``.

            ..  code-block:: python

                class SomeClass:
                    @fnx.module(is_unbound_method=True)
                    def __call__(self, x):
                        ...

    Returns:
        The fnx module.
    """
    if func is None:
        return partial(module, name=name, is_unbound_method=is_unbound_method)

    if is_unbound_method:
        # Method is unbound: Wait until it is called with self instance to create Module
        def outer_wrapper(self, *args, **kwargs):
            @module(name=name)
            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):
                return func(self, *args, **kwargs)
            return inner_wrapper(*args, **kwargs)
        return outer_wrapper

    if name is None:
        class_name = func.__name__
    else:
        class_name = name

    @functools.wraps(func)
    def wrapper(*args, name, skip_module=False, **kwargs):
        if skip_module:
            return func(*args, **kwargs)
        class Module(hk.Module):
            def __init__(self, name=None):
                super().__init__(name=name)

            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)
        Module.__name__ = class_name

        return partial(Module(name=name), *args, **kwargs)

    wrapper = wrap_module(wrapper, name=name)

    return wrapper

class Shared:
    def __init__(self, func):
        self.func = func
        self.module = None
        self.reap = None

    def __call__(self, *args, reap=None, **kwargs):
        if self.module is None:
            self.reap = reap
            result, self.module = fnx.module(self.func, name="shared")(*args, return_module=True, reap=reap, **kwargs)
            return result
        else:
            if self.reap != reap:
                raise ValueError("fnx.shared must always be called with the same reap argument")
            return self.module(*args, **kwargs)

def shared(func):
    return Shared(func)

def sow(name, value, prefix=None):
    """Sows a value with the given name to be reaped by ``fnx.Reaper``.

    Parameters:
        name: Name to which the value is sowed.
        value: The sowed value.
        prefix: Scope that is prefixed to ``name``. Defaults to ``hk.experimental.current_name()``.
    """

    if len(name) == 0:
        raise ValueError("Cannot sow empty name")
    if prefix is None:
        prefix = hk.experimental.current_name()
    fullname = prefix + "/" + name

    for reaper in _reapers():
        reaper.add(fullname, value)
    if fullname.endswith("/~"):
        fullname = fullname[:-2]
        for reaper in _reapers():
            reaper.add(fullname, value)



def init(func, update_fn):
    """Returns a copy of ``func`` where weights are replaced using ``update_fn``.

    Example usage:

    ..  code-block:: python

        @fnx.module
        def model(x):
            ...

        def update_fn(weights):
            # weights: {name: weight}
            weights["name_of_updated_weight"] *= 10.0
            return weights

        model = fnx.init(model, update_fn)

    Parameters:
        func: Function where weights should be replaced.
        update_fn: Function that updates the weights.

    Returns:
        A copy of ``func`` with weights replaced via ``update_fn``.
    """

    def wrapper(*args, **kwargs):
        parent_name = hk.experimental.current_name() + "/"

        values = {}
        def creator(next_creator, shape, dtype, init, context, is_param):
            values[context.full_name] = (shape, dtype, is_param)
            return next_creator(shape, dtype, init)
        def getter(next_getter, value, context, is_param):
            values[context.full_name] = (value.shape, value.dtype, is_param)
            return next_getter(value)
        def setter(next_setter, value, context):
            values[context.full_name] = (value.shape, value.dtype, False)
            return next_setter(value)

        with hk.custom_creator(partial(creator, is_param=True), params=True, state=False), hk.custom_creator(partial(creator, is_param=False), params=False, state=True):
            with hk.custom_getter(partial(getter, is_param=True), params=True, state=False), hk.custom_getter(partial(getter, is_param=False), params=False, state=True):
                with hk.custom_setter(setter):
                    result = func(*args, **kwargs)

        if parent_name == "~/":
            parent_name = ""

        assert all(k.startswith(parent_name) for k in values)
        values = {k[len(parent_name):]: v for k, v in values.items()}

        if hk.running_init():
            frame = hk_base.current_frame()

            updated_values = update_fn(values)

            for name, update_value in updated_values.items():
                shape, dtype, is_param = values[name]

                if not isinstance(update_value, jnp.ndarray):
                    update_value = np.asarray(update_value)
                if not is_param:
                    update_value = hk_base.StatePair(update_value, update_value)

                name = name.split("/")
                prefix = parent_name + "/".join(name[:-1])
                name = name[-1]

                dest = frame.params if is_param else frame.state
                assert prefix in dest and name in dest[prefix]
                dest[prefix][name] = update_value

        return result

    if "__name__" in dir(func):
        name = func.__name__
    else:
        name = str(type(func))
    wrapper.__name__ = name

    return wrapper
