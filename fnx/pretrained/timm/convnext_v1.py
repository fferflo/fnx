import os, fnx, jax, re
import haiku as hk
from functools import partial
import jax.numpy as jnp
import numpy as np

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
    fnx.intercept.defaults(fnx.scale, init=1e-6),
)

class Builder:
    def __init__(self, variant, channels, url):
        self.encoder = vars(fnx.convnext)[f"convnext_{variant}"]
        self.url = url
        self.channels = channels
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                x = self.encoder(x, block=fnx.convnext.block_v1)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=(1, 2))
                    x = fnx.norm(x)
                    x = fnx.linear(x, channels=self.channels, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        model_fn = fnx.pretrained.weights.init(model_fn, weights)

        return model_fn(x)

convnext_v1_atto_imagenet1k_224 = Builder("atto", 1000, "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth")
convnext_v1_femto_imagenet1k_224 = Builder("femto", 1000, "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth")
convnext_v1_pico_imagenet1k_224 = Builder("pico", 1000, "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth")
convnext_v1_nano_imagenet1k_224 = Builder("nano", 1000, "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth")

convnext_v1_tiny_imagenet1k_288 = Builder("tiny", 1000, "https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k/blob/main/pytorch_model.bin")
convnext_v1_small_imagenet1k_288 = Builder("small", 1000, "https://huggingface.co/timm/convnext_small.in12k_ft_in1k/blob/main/pytorch_model.bin")

convnext_v1_nano_imagenet12k_224 = Builder("nano", 11821, "https://huggingface.co/timm/convnext_nano.in12k/blob/main/pytorch_model.bin")
convnext_v1_tiny_imagenet12k_224 = Builder("tiny", 11821, "https://huggingface.co/timm/convnext_tiny.in12k/blob/main/pytorch_model.bin")
convnext_v1_small_imagenet12k_224 = Builder("small", 11821, "https://huggingface.co/timm/convnext_small.in12k/blob/main/pytorch_model.bin")
