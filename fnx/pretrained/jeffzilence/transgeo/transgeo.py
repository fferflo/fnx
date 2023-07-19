import os, fnx, jax, re, pyunpack, shutil
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
    def __init__(self, name, gid):
        self.name = name
        self.gid = gid
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x, mode="aerial"):
        if mode == "reference":
            mode = ["reference", "query"]
            pos_embed_shape = (20, 20)
        elif mode == "query":
            mode = ["query", "reference"]
            pos_embed_shape = (7, 38)
        else:
            raise ValueError(f"Invalid mode {mode}")

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
                    x = fnx.vit.vit_small(x, pos_embed_shape=pos_embed_shape, patch_size=16, prefix_tokens=2)
                with fnx.scope("head"):
                    x0 = fnx.linear(x[:, 0], channels=1000, bias=True, name="embed1")
                    x1 = fnx.linear(x[:, 1], channels=1000, bias=True, name="embed2")
                    x = 0.5 * (x0 + x1)
            return x

        file = os.path.join(fnx.pretrained.weights.path, self.name + ".pth.tar")
        if not os.path.isfile(file):
            weights_compressed = fnx.pretrained.weights.download_googledrive(self.name + ".tar.gz", self.gid)
            folder = file[:-8]
            if not os.path.isdir(folder):
                os.makedirs(folder)
            pyunpack.Archive(weights_compressed).extractall(folder)
            shutil.move(os.path.join(folder, "result", "model_best.pth.tar"), file)
            shutil.rmtree(folder)
            os.remove(weights_compressed)
        weights = fnx.pretrained.weights.load_pytorch(file)

        # 3. They add additional positional embedding onto prefix tokens, we add positional embedding only on image tokens
        pos_embed = weights[f"module.{mode[0]}_net.pos_embed"]
        cls_token = weights[f"module.{mode[0]}_net.cls_token"]
        dist_token = weights[f"module.{mode[0]}_net.dist_token"]
        del weights[f"module.{mode[0]}_net.cls_token"]
        del weights[f"module.{mode[0]}_net.dist_token"]
        weights[f"module.{mode[0]}_net.prefix_tokens"] = jnp.concatenate([cls_token, dist_token], axis=1) + pos_embed[:, :2, :]
        weights[f"module.{mode[0]}_net.pos_embed"] = pos_embed[:, 2:, :].reshape((1, pos_embed_shape[0], pos_embed_shape[1], pos_embed.shape[-1]))

        for k in list(weights.keys()):
            if mode[1] in k:
                del weights[k]

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("/attn/norm", ".norm1"), ("/head/embed2", ".head_dist")])

        return model_fn(x)

transgeo_cvusa = Builder("transgeo_cvusa", "1D4Jk8Mho82YvPhxlNwiOuDhIAKLGp_-7")
transgeo_cvact = Builder("transgeo_cvact", "1cU3qTi343Yl1b7tdUoprseQpQ6IZ5oLs")
transgeo_vigor = Builder("transgeo_vigor", "1NzeYlKXPjr59ERWiC38YT8Z7ZbHMkBNe")
