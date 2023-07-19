import threading, contextlib, functools
import haiku as hk

_thread_local = threading.local()
def _interceptors():
    if not hasattr(_thread_local, "interceptors"):
        _thread_local.interceptors = []
    return _thread_local.interceptors

def wrap(func, context):
    old_func = func
    def run_interceptors(func, args, kwargs, interceptors, context):
        if len(interceptors) > 0:
            interceptor = interceptors[-1]
            def next_interceptor(func, args, kwargs):
                return run_interceptors(func, args, kwargs, interceptors[:-1], context)
            return interceptor(next_interceptor, func, args, kwargs, context)
        else:
            return func(*args, **kwargs)

    _thread_local = threading.local()
    @functools.wraps(old_func)
    def new_func(*args, **kwargs):
        if not "should_intercept" in vars(_thread_local):
            _thread_local.should_intercept = True
        if _thread_local.should_intercept:
            _thread_local.should_intercept = False
            result = run_interceptors(new_func, args, kwargs, _interceptors(), context(*args, **kwargs))
            _thread_local.should_intercept = True
            return result
        else:
            return old_func(*args, **kwargs)

    return new_func

class custom:
    """A module interceptor which modifies calls to modules out-of-place. Calls the passed function ``interceptor`` before every call to a fnx module.

    For example, to remove all calls to ``fnx.layernorm``:

    ..  code-block:: python

        # Define a custom model
        def model(x):
            x = fnx.layernorm(x)
            return x

        # Define an interceptor to remove layernorm
        def interceptor(next_interceptor, module, args, kwargs, context: fnx.ModuleCallContext):
            # We are currently before the call module(*args, **kwargs)
            if module == fnx.layernorm:
                # Calling fnx.layernorm next:
                # Stop here, and return the original input instead
                return args[0]
            else:
                # Otherwise:
                # Call the next outer interceptor
                # (-> module itself if no more interceptors are specified)
                return next_interceptor(func, args, kwargs)

        # Call the model without layernorm
        with fnx.intercept.custom(interceptor):
            x = model(x)

    Parameters:
        interceptor: Function that is called before every call to a fnx module.
    """

    def __init__(self, interceptor):
        self.interceptor = interceptor

    def __enter__(self):
        _interceptors().append(self.interceptor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _interceptors().pop()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper

def defaults(module, **kwargs):
    """Set default arguments for calls to the given module.

    Parameters:
        module: The module to be replaced.
        **kwargs: New default arguments for the module.

    Returns:
        A context manager.
    """
    def interceptor(next_interceptor, func, args, ikwargs, context):
        if module == func:
            for k, v in kwargs.items():
                if not k in ikwargs:
                    ikwargs[k] = v
        return next_interceptor(func, args, ikwargs)
    return custom(interceptor)

def replace(old, new):
    """Replace calls to the given module with another module.

    Parameters:
        old: The replaced module.
        new: The replacing module.

    Returns:
        A context manager.
    """
    def interceptor(next_interceptor, func, args, kwargs, context):
        if func == old:
            func = new
        return next_interceptor(func, args, kwargs)
    return custom(interceptor)

def remove_if(pred):
    """Remove module calls determined by ``pred``.

    Parameters:
        pred: A function ``fn: func, args, kwargs, context -> bool`` that returns ``True`` if a given module call should be removed and ``False`` otherwise.

    Returns:
        A context manager.

    Note:
        Works only with modules that return a single result.
    """
    def interceptor(next_interceptor, func, args, kwargs, context):
        if pred(func, args, kwargs, context):
            return args[0]
        return next_interceptor(func, args, kwargs)
    return custom(interceptor)

def remove(module):
    """Removes calls to the given module.

    Parameters:
        module: The removed module.

    Returns:
        A context manager.

    Note:
        Works only with modules that return a single result.
    """
    return remove_if(lambda func, args, kwargs, context: func == old)

stop = custom(lambda next_interceptor, func, args, kwargs, context: func(*args, **kwargs))
"""Stops outer interceptors from modifying inner module calls.

Example usage:

..  code-block:: python

    def model(x):
        with fnx.intercept.stop: # Stop outer interceptors from modifying inner calls
            x = fnx.layernorm(x) # This is not replaced
        return x

    with fnx.intercept.replace(fnx.layernorm, fnx.batchnorm):
        x = model(x)

"""

class chain:
    """Chains the given context managers.

    For example, the following two code snippets are equivalent:

    ..  code-block:: python

        with fnx.intercept.replace(fnx.norm, fnx.batchnorm):
            with fnx.intercept.replace(fnx.act, jax.nn.relu):
                x = model(x)

    ..  code-block:: python

        config = fnx.intercept.chain(
            fnx.intercept.replace(fnx.norm, fnx.batchnorm),
            fnx.intercept.replace(fnx.act, jax.nn.relu),
        )
        with config:
            x = model(x)

    Parameters:
        *args: The list of context managers
    """
    def __init__(self, *args):
        self.managers = args

    def __enter__(self):
        for m in self.managers:
            m.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in reversed(self.managers):
            m.__exit__(exc_type, exc_val, exc_tb)
