import os, fnx, jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import haiku as hk

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.swish),
    fnx.config.tensorflow,
)

class Builder:
    def __init__(self, variant, resolution, url):
        self.encoder = vars(fnx.efficientnet)[f"efficientnet_{variant}"]
        self.resolution = resolution
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                x = self.encoder(x)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=(1, 2))
                    x = fnx.linear(x, channels=1000, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("/expand/norm", ".bn1"), ("/depthwise/norm", ".bn1")])

        return model_fn(x)

efficientnet_b0_imagenet1k = Builder("b0", (224, 224), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth")
efficientnet_b1_imagenet1k = Builder("b1", (240, 240), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth")
efficientnet_b2_imagenet1k = Builder("b2", (260, 260), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth")
efficientnet_b3_imagenet1k = Builder("b3", (300, 300), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth")
efficientnet_b4_imagenet1k = Builder("b4", (380, 380), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth")
efficientnet_b5_imagenet1k = Builder("b5", (456, 456), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth")
efficientnet_b6_imagenet1k = Builder("b6", (528, 528), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth")
efficientnet_b7_imagenet1k = Builder("b7", (600, 600), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth")
efficientnet_b8_imagenet1k = Builder("b8", (672, 672), "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth")
