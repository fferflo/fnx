import jax.numpy as jnp
import fnx, jax
import haiku as hk
import numpy as np

# https://arxiv.org/abs/1603.09382
# scale_at_train_time=True: implemented as in https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
# scale_at_train_time=False: implemented as in tfa.layers.StochasticDepth
@fnx.module
def drop(x, axis, drop_rate=0.0, scale_at_train_time=True):
    if drop_rate >= 1.0 or drop_rate < 0.0:
        raise ValueError(f"Invalid drop_rate, must be in range [0.0, 1.0)")

    drop_rate = np.float16(drop_rate)

    if scale_at_train_time and fnx.is_training:
        x = x / (1 - drop_rate)
    elif not scale_at_train_time and not fnx.is_training:
        x = x * (1 - drop_rate)

    if fnx.is_training and drop_rate > 0:
        shape = np.ones_like(x.shape)
        if isinstance(axis, int):
            axis = (axis,)
        for a in axis:
            shape[a] = x.shape[a]
        drop = jax.random.bernoulli(hk.next_rng_key(), drop_rate, shape=shape)
        x = jnp.where(drop, 0.0, x)

    return x

# Drops entire instances in batch
def droppath(x, *args, **kwargs):
    return drop(x, 0, *args, **kwargs)

# Independently drops values in tensor
def dropout(x, *args, **kwargs):
    return drop(x, list(range(len(x.shape))), *args, **kwargs)
