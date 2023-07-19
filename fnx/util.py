import fnx, jax, threading, uuid, subprocess, os, time
import jax.numpy as jnp
import numpy as np
import haiku as hk
from collections import defaultdict

_thread_local = threading.local()
def _stack():
    if not hasattr(_thread_local, "is_training"):
        _thread_local.is_training = []
    return _thread_local.is_training

def is_training():
    if not hasattr(_thread_local, "is_training") or len(_thread_local.is_training) == 0:
        raise ValueError("No value set for is_training")
    else:
        return _thread_local.is_training[-1]

class set_is_training:
    def __init__(self, is_training):
        self.is_training = is_training

    def __enter__(self):
        _stack().append(self.is_training)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _stack().pop()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper

def replicate(x, rank):
    if isinstance(x, int) or isinstance(x, np.int32) or isinstance(x, np.int64):
        return (int(x),) * rank
    else:
        x = np.asarray(x)
        if x.shape != (rank,):
            raise ValueError(f"Invalid shape")
        return tuple(x.tolist())

def flatten(tree, separator="//"):
    result = {}
    for ok, ov in tree.items():
        if isinstance(ov, dict):
            for ik, iv in flatten(ov, separator=separator).items():
                result[ok + separator + ik] = iv
        else:
            result[ok] = ov
    return result

def unflatten(values, separator="//"):
    result = {}
    for name, value in values.items():
        keys = name.split(separator)
        node = result
        for k in keys[:-1]:
            if not k in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return result

def subshape(shape, axis):
    """Returns a new shape of the same rank as ``shape`` that is equal to ``shape`` for all ``axis`` and 1 elsewhere.

    Args:
        shape: Sequence[int],
        axis: Union[int, Sequence[int]]

    Returns:
        The subshape of ``shape``.

    Examples:
        >>> subshape([4, 128, 128, 32], axis=-1)
        (1, 1, 1, 32)

        >>> subshape([4, 128, 128, 32], axis=(0, 2))
        (4, 1, 128, 1)
    """

    if isinstance(axis, int):
        axis = (axis,)
    out_shape = [1] * len(shape)
    for a in axis:
        out_shape[a] = shape[a]
    return tuple(out_shape)

# def find_unused(func, *args, static_argnames=[], **kwargs):
#     rng = hk.PRNGSequence(42)
#     model = hk.transform_with_state(func)
#     params, state = model.init(next(rng), *args, **kwargs)
#     names = sorted(list(flatten(params).keys())) + sorted(list(flatten(state).keys()))
#     def wrapper(*args, **kwargs):
#         static_kwargs = {}
#         dynamic_args = [params, state, next(rng), *args]
#         dynamic_kwargs = {}
#         for k, v in kwargs.items():
#             if k in static_argnames:
#                 static_kwargs[k] = v
#             else:
#                 dynamic_kwargs[k] = v
#
#         # Create jaxpr
#         jaxpr = jax.make_jaxpr(lambda *args, **kwargs: model.apply(*args, **static_kwargs, **kwargs)[0])(*dynamic_args, **dynamic_kwargs)
#
#         # Parse equations
#         outvar_to_invars = defaultdict(list)
#         for eqn in jaxpr.jaxpr.eqns:
#             # print(f"IN: {[v.count for v in eqn.invars if isinstance(v, jax.core.Var)]}")
#             # print(f"OUT: {[v.count for v in eqn.outvars if isinstance(v, jax.core.Var)]}")
#             # print(f"EQN:\n{eqn}")
#             # print()
#             for invar in eqn.invars:
#                 for outvar in eqn.outvars:
#                     if isinstance(invar, jax.core.Var) and isinstance(outvar, jax.core.Var):
#                         outvar_to_invars[outvar.count].append(invar.count)
#
#         invars = set(v.count for v in jaxpr.jaxpr.invars)
#         outvars = set(v.count for v in jaxpr.jaxpr.outvars)
#
#         # print(f"constvars={set(v.count for v in jaxpr.jaxpr.constvars)}")
#         # print(f"invars={invars}")
#         # print(f"outvars={outvars}")
#
#         # Find unused inputs
#         def find_recursive_invars(todo):
#             used_vars = set()
#             todo = [v for v in todo]
#             while len(todo) > 0:
#                 v = todo[0]
#                 todo = todo[1:]
#                 if not v in used_vars:
#                     used_vars.add(v)
#                     todo.extend(outvar_to_invars[v])
#             return used_vars
#         vars_used_for_output = find_recursive_invars(outvars)
#         invars_used_for_output = vars_used_for_output.intersection(invars)
#         invars_unused_for_output = invars.difference(invars_used_for_output)
#         # print(f"invars_unused_for_output={invars_unused_for_output}")
#
#         # Convert to names
#         constvars_offset = len(jaxpr.jaxpr.constvars)
#         unused_names = [i - constvars_offset for i in invars_unused_for_output]
#         unused_names = [names[i] for i in unused_names if i < len(names)]
#         return unused_names
#
#     vars(wrapper)["params"] = params
#     vars(wrapper)["state"] = state
#     vars(wrapper)["names"] = names
#     return wrapper

def remove_unused(func, names):
    if isinstance(names, list) or isinstance(names, tuple):
        names = lambda n: n in names
    def wrapper(*args, **kwargs):
        dummy_values = {}
        def creator(next_creator, shape, dtype, init, context):
            if names(context.full_name):
                dummy_values[context.full_name] = jnp.zeros(shape, dtype)
                return hk.experimental.DO_NOT_STORE
            else:
                return next_creator(shape, dtype, init)

        def getter(next_getter, value, context):
            if names(context.full_name):
                return dummy_values[context.full_name]
            else:
                return next_getter(value)

        with hk.custom_creator(creator, params=True, state=True), hk.custom_getter(getter, params=True, state=True):
            return func(*args, **kwargs)
    return wrapper

class MemoryTracker:
    def __init__(self, period=0.1):
        self.lock = threading.Lock()
        self.peak_usage = None
        self.period = period

        def run():
            path = f"/dev/shm/fnx-mem-{uuid.uuid4().hex}.prof"
            while True:
                jax.profiler.save_device_memory_profile(path)

                lines = subprocess.run(
                    args=f"go tool pprof -tags {path}".split(" "),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                ).stdout.decode("utf-8")

                lines = lines.split("\n")
                lines = lines[1:lines.index("")]
                lines = [l.strip() for l in lines]

                usage = {}
                for l in lines:
                    l = l.split(" ")
                    device = l[-1]
                    if l[0].endswith("GB"):
                        num_bytes = float(l[0][:-2]) * (1024 ** 3)
                    elif l[0].endswith("MB"):
                        num_bytes = float(l[0][:-2]) * (1024 ** 2)
                    elif l[0].endswith("KB"):
                        num_bytes = float(l[0][:-2]) * (1024 ** 1)
                    else:
                        assert l[0].endswith("B") and not l[0][-2].isalpha()
                        num_bytes = float(l[0][:-1])
                    usage[device] = num_bytes

                with self.lock:
                    if self.peak_usage is None:
                        self.peak_usage = usage
                    else:
                        for k, v in usage.items():
                            if k in self.peak_usage:
                                self.peak_usage[k] = max(v, self.peak_usage[k])
                            else:
                                self.peak_usage[k] = v


                time.sleep(period)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def get_peak_usage(self):
        while self.peak_usage is None:
            time.sleep(self.period)
        with self.lock:
            result = {**self.peak_usage}
            self.peak_usage = None
        return result
