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
)

class Builder:
    def __init__(self, variant, patch_size, pos_embed_shape, url):
        self.encoder = vars(fnx.vit)[f"vit_{variant}"]
        self.patch_size = patch_size
        self.pos_embed_shape = pos_embed_shape
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            def replace(next_interceptor, func, args, kwargs, context):
                # 1. They don't use layerscale
                # 2. They don't have normalization in patch embedding
                if context.fullname.endswith("scale") or context.fullname.endswith("patch_embed/norm") or context.fullname.endswith("encode/norm"):
                    return args[0]
                else:
                    return next_interceptor(func, args, kwargs)

            with config:
                with fnx.intercept.custom(replace):
                    x = self.encoder(x, pos_embed_shape=self.pos_embed_shape, patch_size=self.patch_size, prefix_tokens=1)
                with fnx.scope("classifier"):
                    x = jnp.mean(x[:, 1:], axis=1)
                    x = fnx.norm(x)
                    x = fnx.linear(x, channels=1000, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)

        # 3. They add an additional positional embedding onto class token, we add positional embedding only on image tokens
        pos_embed = weights["pos_embed"]
        cls_token = weights["cls_token"]
        weights["cls_token"] = cls_token + pos_embed[:, :1, :]
        weights["pos_embed"] = pos_embed[:, 1:, :].reshape((1, self.pos_embed_shape[0], self.pos_embed_shape[1], pos_embed.shape[-1]))

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "norm1")])

        return model_fn(x)

vit_large_patch14_196_eva_in22k_ft_in22k_in1k = Builder("large", 14, (14, 14), "https://huggingface.co/timm/eva_large_patch14_196.in22k_ft_in22k_in1k/blob/main/pytorch_model.bin")
vit_large_patch14_336_eva_in22k_ft_in22k_in1k = Builder("large", 14, (24, 24), "https://huggingface.co/timm/eva_large_patch14_336.in22k_ft_in22k_in1k/blob/main/pytorch_model.bin")

vit_large_patch14_196_eva_in22k_ft_in1k = Builder("large", 14, (14, 14), "https://huggingface.co/timm/eva_large_patch14_196.in22k_ft_in1k/blob/main/pytorch_model.bin")
vit_large_patch14_336_eva_in22k_ft_in1k = Builder("large", 14, (24, 24), "https://huggingface.co/timm/eva_large_patch14_336.in22k_ft_in1k/blob/main/pytorch_model.bin")
