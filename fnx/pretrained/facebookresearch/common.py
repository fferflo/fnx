import fnx, jax
import numpy as np
from functools import partial

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.pytorch,
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-6),
    fnx.intercept.defaults(fnx.scale, init=1e-6),
)
