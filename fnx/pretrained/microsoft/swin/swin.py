import os, fnx, jax, re, gdown
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
)

class Builder:
    def __init__(self, variant, num_classes, url):
        self.encoder = vars(fnx.swin)[f"swin_{variant}"]
        self.url = url
        self.num_classes = num_classes
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            input_shape = x.shape
            with config:
                x = self.encoder(x)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=(1, 2))
                    x = fnx.linear(x, channels=self.num_classes, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        weights = {k: v for k, v in weights.items() if not "relative_position_index" in k and not "attn_mask" in k}
        for k in list(weights.keys()):
            if "relative_position_bias_table" in k:
                # 1. They store prior logits in spatially flattened array
                v = weights[k]
                w = int(np.sqrt(v.shape[0]))
                weights[k] = v.reshape([w, w, v.shape[-1]])
            elif "downsample.reduction.weight" in k:
                # 2. They don't downsample in row-major format
                v = weights[k] # (c_in0 + c_in2 + c_in1 + c_in3) c_out
                v = np.split(v, axis=0, indices_or_sections=4)
                v[2], v[1] = v[1], v[2]
                v = np.concatenate(v, axis=0)
                weights[k] = v # (c_in0 + c_in1 + c_in2 + c_in3) c_out

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "norm1")])

        return model_fn(x)

swin_tiny_imagenet1k_224 = Builder("tiny", 1000, "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth")
swin_small_imagenet1k_224 = Builder("small", 1000, "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth")
swin_base_imagenet1k_224 = Builder("base", 1000, "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth")

swin_tiny_imagenet22k_224 = Builder("tiny", 21841, "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth")
swin_small_imagenet22k_224 = Builder("small", 21841, "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth")
swin_base_imagenet22k_224 = Builder("base", 21841, "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth")
swin_large_imagenet22k_224 = Builder("large", 21841, "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth")
