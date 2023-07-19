import os, fnx, jax
import haiku as hk
from functools import partial
import jax.numpy as jnp

from ..common import color_mean, color_std, preprocess, config

class Builder:
    def __init__(self, variant, url):
        self.encoder = vars(fnx.convnext)[f"convnext_{variant}"]
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                x = self.encoder(x, block=fnx.convnext.block_v1)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=(1, 2))
                    x = fnx.norm(x)
                    x = fnx.linear(x, channels=1000 if "1k" in self.url else 21841, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        model_fn = fnx.pretrained.weights.init(model_fn, weights)

        return model_fn(x)

convnext_v1_tiny_imagenet1k_224 = Builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth")
convnext_v1_small_imagenet1k_224 = Builder("small", f"https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth")
convnext_v1_base_imagenet1k_224 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth")
convnext_v1_large_imagenet1k_224 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth")

convnext_v1_base_imagenet1k_384 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth")
convnext_v1_large_imagenet1k_384 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth")

convnext_v1_base_imagenet22k_224 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth")
convnext_v1_large_imagenet22k_224 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth")
convnext_v1_xlarge_imagenet22k_224 = Builder("xlarge", f"https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth")
