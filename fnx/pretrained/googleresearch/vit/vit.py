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
                if context.fullname.endswith("scale") or context.fullname.endswith("patch_embed/norm"):
                    return args[0]
                else:
                    return next_interceptor(func, args, kwargs)

            with config:
                with fnx.intercept.custom(replace):
                    x = self.encoder(x, pos_embed_shape=self.pos_embed_shape, patch_size=self.patch_size, prefix_tokens=1)
                with fnx.scope("classifier"):
                    x = x[:, 0]
                    x = fnx.linear(x, channels=21843, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_numpy(file)

        # 3. They add an additional positional embedding onto class token, we add positional embedding only on image tokens
        pos_embed = weights["Transformer/posembed_input/pos_embedding"]
        cls_token = weights["cls"]
        weights["cls"] = cls_token + pos_embed[:, :1, :]
        weights["Transformer/posembed_input/pos_embedding"] = pos_embed[:, 1:, :].reshape((1, self.pos_embed_shape[0], self.pos_embed_shape[1], pos_embed.shape[-1]))

        # 4. They use three different linear layers for query-key-value, we use a single layer
        # 5. They split head and channel dimensions, we don't
        i = 0
        while True:
            prefix = f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1"
            if not f"{prefix}/key/bias" in weights:
                break

            def reshape(name):
                x = weights[name]
                del weights[name]
                return x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
            weights[f"{prefix}/qkv/kernel"] = np.concatenate([
                reshape(f"{prefix}/query/kernel"),
                reshape(f"{prefix}/key/kernel"),
                reshape(f"{prefix}/value/kernel"),
            ], axis=1)

            def reshape(name):
                x = weights[name]
                del weights[name]
                return x.reshape([x.shape[0] * x.shape[1]])
            weights[f"{prefix}/qkv/bias"] = np.concatenate([
                reshape(f"{prefix}/query/bias"),
                reshape(f"{prefix}/key/bias"),
                reshape(f"{prefix}/value/bias"),
            ], axis=0)

            reshape = lambda x: x.reshape([x.shape[0] * x.shape[1], x.shape[2]])
            weights[f"{prefix}/out/kernel"] = reshape(weights[f"{prefix}/out/kernel"])

            i += 1

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "LayerNorm_0")])

        return model_fn(x)

vit_tiny_patch16_imagenet22k_224 = Builder("tiny", 16, (14, 14), "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz")
vit_small_patch16_imagenet22k_224 = Builder("small", 16, (14, 14), "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz")
vit_base_patch16_imagenet22k_224 = Builder("base", 16, (14, 14), "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz")
vit_large_patch16_imagenet22k_224 = Builder("large", 16, (14, 14), "https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz")

vit_small_patch32_imagenet22k_224 = Builder("small", 32, (7, 7), "https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz")
vit_base_patch32_imagenet22k_224 = Builder("base", 32, (7, 7), "https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz")
