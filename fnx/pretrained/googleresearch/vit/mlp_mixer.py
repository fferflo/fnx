import os, fnx, jax, re
import haiku as hk
from functools import partial
import jax.numpy as jnp
import numpy as np

def preprocess(color):
    return (color - 127.5) / 127.5

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.flax,
)

class Builder:
    def __init__(self, variant, patch_size, classes, url):
        self.encoder = vars(fnx.mlp_mixer)[f"mlp_mixer_{variant}"]
        self.patch_size = patch_size
        self.classes = classes
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                # 1. They don't use layerscale
                # 2. They don't have normalization in patch embedding
                with fnx.intercept.remove_if(lambda func, args, kwargs, context: context.fullname.endswith("scale") or context.fullname.endswith("patch_embed/norm")): # TODO: add remove_if with lambda context: (check param names and nums)
                    x = self.encoder(x, patch_size=self.patch_size)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=1)
                    x = fnx.linear(x, channels=self.classes, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_numpy(file)

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("spatial_mix/norm", "LayerNorm_0")])

        return model_fn(x)

mlp_mixer_base_patch16_imagenet1k_224 = Builder("base", 16, 1000, "https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz")
mlp_mixer_base_patch16_imagenet22k_224 = Builder("base", 16, 21843, "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz")

mlp_mixer_large_patch16_imagenet1k_224 = Builder("large", 16, 1000, "https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-L_16.npz")
mlp_mixer_large_patch16_imagenet22k_224 = Builder("large", 16, 21843, "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-L_16.npz")
