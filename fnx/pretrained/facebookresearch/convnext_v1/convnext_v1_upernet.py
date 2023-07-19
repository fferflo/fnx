import os, fnx, jax, re
import haiku as hk
from functools import partial
import jax.numpy as jnp

from ..common import color_mean, color_std, preprocess
from ..common import config as encoder_config

decoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, variant, url):
        self.encoder = vars(fnx.convnext)[f"convnext_{variant}"]
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            input_shape = x.shape
            with encoder_config:
                xs = self.encoder(x, block=fnx.convnext.block_v1, reap=["stage1", "stage2", "stage3", "stage4"])
                xs = [fnx.norm(x, name=f"norm{i + 1}") for i, x in enumerate(xs)]
            with decoder_config:
                x = fnx.upernet.decode(xs, channels=512, psp_bin_sizes=[1, 2, 3, 6])
                with fnx.scope("classifier"):
                    x = hk.dropout(hk.next_rng_key(), 0.1, x)
                    x = fnx.linear(x, channels=150, bias=True)
                    x = jax.image.resize(x, shape=input_shape[:-1] + (150,), method="bilinear")
                    fnx.sow("logits", x)
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        weights = {k: v for k, v in weights.items() if not k.startswith("aux")}
        model_fn = fnx.pretrained.weights.init(model_fn, weights)

        return model_fn(x)

convnext_v1_tiny_upernet_imagenet1k_ade20k_512 = Builder("tiny", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth")
convnext_v1_small_upernet_imagenet1k_ade20k_512 = Builder("small", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_small_1k_512x512.pth")
convnext_v1_base_upernet_imagenet1k_ade20k_512 = Builder("base", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_base_1k_512x512.pth")

convnext_v1_base_upernet_imagenet22k_ade20k_640 = Builder("base", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_base_22k_640x640.pth")
convnext_v1_large_upernet_imagenet22k_ade20k_640 = Builder("large", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_large_22k_640x640.pth")
convnext_v1_xlarge_upernet_imagenet22k_ade20k_640 = Builder("xlarge", "https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_xlarge_22k_640x640.pth")
