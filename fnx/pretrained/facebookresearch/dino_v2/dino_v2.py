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
    fnx.intercept.defaults(fnx.scale, init=1e-6),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, variant, patch_size, pos_embed_shape, swiglu, url):
        self.encoder = vars(fnx.vit)[f"vit_{variant}"]
        self.patch_size = patch_size
        self.pos_embed_shape = pos_embed_shape
        self.swiglu = swiglu
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            def replace(next_interceptor, func, args, kwargs, context):
                # 1. They don't have normalization in patch embedding
                # 2. For giant2 backbone, they use SwiGLU activation
                if context.fullname.endswith("patch_embed/norm"):
                    return args[0]
                elif self.swiglu and context.fullname.endswith("linear1"):
                    kwargs["channels"] = 2 * kwargs["channels"]
                elif self.swiglu and context.fullname.endswith("act"):
                    x1, x2 = jnp.split(args[0], 2,  axis=-1)
                    return jax.nn.silu(x1) * x2
                return next_interceptor(func, args, kwargs)

            with config:
                with fnx.intercept.custom(replace):
                    x = self.encoder(x, pos_embed_shape=self.pos_embed_shape, patch_size=self.patch_size, prefix_tokens=1)
                with fnx.scope("classifier"):
                    x = jnp.concatenate([
                        x[:, 0],
                        jnp.mean(x[:, 1:], axis=1),
                    ], axis=-1)
                    x = fnx.linear(x, channels=1000, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url + "pretrain.pth")
        weights_backend = fnx.pretrained.weights.load_pytorch(file)
        file = fnx.pretrained.weights.download(self.url + "linear_head.pth")
        weights_classifier = fnx.pretrained.weights.load_pytorch(file)
        weights = weights_backend | weights_classifier
        del weights["mask_token"]

        # 2. They add an additional positional embedding onto class token, we add positional embedding only on image tokens
        pos_embed = weights["pos_embed"]
        cls_token = weights["cls_token"]
        weights["cls_token"] = cls_token + pos_embed[:, :1, :]
        weights["pos_embed"] = pos_embed[:, 1:, :].reshape((1, self.pos_embed_shape[0], self.pos_embed_shape[1], pos_embed.shape[-1]))

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "norm1"), ("attn/scale", "ls1")])

        return model_fn(x)

dino_v2_small_patch14_imagenet1k = Builder("small", 14, (37, 37), False, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_")
dino_v2_base_patch14_imagenet1k = Builder("base", 14, (37, 37), False, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_")
dino_v2_large_patch14_imagenet1k = Builder("large", 14, (37, 37), False, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_")
dino_v2_small_patch14_imagenet1k = Builder("giant2", 14, (37, 37), True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_")
