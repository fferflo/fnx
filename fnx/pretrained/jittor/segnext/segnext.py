import os, fnx, jax, re, gdown
import haiku as hk
from functools import partial
import jax.numpy as jnp
import numpy as np

color_mean = np.asarray([123.675, 116.28, 103.53])
color_std = np.asarray([58.395, 57.12, 57.375])
def preprocess(color):
    color = (color - color_mean) / color_std
    return color

encoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.intercept.defaults(fnx.scale, init=1e-2),
    fnx.config.jittor,
)

decoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.groupnorm),
    fnx.intercept.replace(fnx.act, jax.nn.relu),
    fnx.config.jittor,
    fnx.intercept.defaults(fnx.groupnorm, groups=32),
)

class Builder:
    def __init__(self, variant, bases_num, decode_channels, num_classes, url):
        self.encoder = vars(fnx.segnext)[f"mscan_{variant}"]
        self.decode_channels = decode_channels
        self.url = url
        self.bases_num = bases_num
        self.num_classes = num_classes
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            input_shape = x.shape

            # 1. They use batchnorm in blocks and layernorm at the end of stages
            def replace_norm(next_interceptor, func, args, kwargs, context):
                if re.search("stage[0-9]\\/norm", context.fullname):
                    return next_interceptor(fnx.layernorm, [args[0]], {})
                else:
                    return next_interceptor(func, args, kwargs)

            with encoder_config, fnx.intercept.custom(replace_norm):
                xs = self.encoder(x, reap=["stage2", "stage3", "stage4"])
            with decoder_config:
                x = fnx.hamburger.decode(xs, channels=self.decode_channels, matrix_decomposition=partial(fnx.hamburger.nonnegative_matrix_factorization, bases_num=self.bases_num, iterations=6 if fnx.is_training else 7))
                with fnx.scope("classifier"):
                    x = hk.dropout(hk.next_rng_key(), 0.1, x)
                    x = fnx.linear(x, channels=self.num_classes, bias=True)
                    x = jax.image.resize(x, shape=input_shape[:-1] + (self.num_classes,), method="bilinear")
                    fnx.sow("logits", x)
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_jittor(file)
        model_fn = fnx.pretrained.weights.init(model_fn, weights, ignore=lambda n: n.endswith("bases"), hints=[("/attn/scale/scale", ".layer_scale_1"), ("/mlp/scale/scale", ".layer_scale_2"), ("/attn/norm", ".norm1")])

        return model_fn(x)

segnext_tiny_cityscapes_1024 = Builder("tiny", 16, 256, 19, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_tiny_1024x1024_city_160k.pkl")
segnext_small_cityscapes_1024 = Builder("small", 16, 256, 19, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_small_1024x1024_city_160k.pkl")
segnext_base_cityscapes_1024 = Builder("base", 64, 512, 19, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_base_1024x1024_city_160k.pkl")
segnext_large_cityscapes_1024 = Builder("large", 64, 1024, 19, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_large_1024x1024_city_160k.pkl")

segnext_tiny_ade20k_512 = Builder("tiny", 16, 256, 150, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_tiny_512x512_ade_160k.pkl")
segnext_small_ade20k_512 = Builder("small", 16, 256, 150, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_small_512x512_ade_160k.pkl")
segnext_base_ade20k_512 = Builder("base", 64, 512, 150, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_base_512x512_ade_160k.pkl")
segnext_large_ade20k_512 = Builder("large", 64, 1024, 150, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_large_512x512_ade_160k.pkl")

segnext_tiny_isaid_512 = Builder("tiny", 16, 256, 16, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_tiny_896x896_isaid_160k.pkl")
segnext_small_isaid_512 = Builder("small", 16, 256, 16, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_small_896x896_isaid_160k.pkl")
segnext_base_isaid_512 = Builder("base", 64, 512, 16, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_base_896x896_isaid_160k.pkl")
segnext_large_isaid_512 = Builder("large", 64, 1024, 16, "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/segnext_large_896x896_isaid_160k.pkl")
