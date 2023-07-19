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
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.pytorch,
)

decoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.relu),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, variant, num_classes, googledrive_id, googledrive_name):
        self.encoder = vars(fnx.segformer)[f"mit_{variant}"]
        self.decode_channels = 256 if int(variant[1:]) <= 1 else 768
        self.googledrive_id = googledrive_id
        self.googledrive_name = googledrive_name
        self.num_classes = num_classes
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            input_shape = x.shape
            with encoder_config:
                xs = self.encoder(x, reap=["stage1", "stage2", "stage3", "stage4"])
            with decoder_config:
                x = fnx.segformer.decode(xs, channels=self.decode_channels)
                with fnx.scope("classifier"):
                    x = hk.dropout(hk.next_rng_key(), 0.1, x) # TODO: use fnx.stochastic
                    x = fnx.linear(x, channels=self.num_classes, bias=True)
                    x = jax.image.resize(x, shape=input_shape[:-1] + (self.num_classes,), method="bilinear")
                    fnx.sow("logits", x)
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download_googledrive(self.googledrive_name, self.googledrive_id)
        weights = fnx.pretrained.weights.load_pytorch(file)
        weights = {k: v for k, v in weights.items() if not "decode_head.conv_seg" in k}
        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "norm1"), ("/query/", ".q.")])

        return model_fn(x)

segformer_b0_ade20k_512 = Builder("b0", 150, "1je1GL6TXU3U-cZZsUv08ITUkVW4mBPYy", "segformer.b0.512x512.ade.160k.pth")
segformer_b0_cityscapes_512x1024 = Builder("b0", 19, "1yjPTULZCGAYpK0XCg1SNHXhMga4_w03r", "segformer.b0.512x1024.city.160k.pth")
segformer_b0_cityscapes_640x1024 = Builder("b0", 19, "1t4fOtwJqpnUJvMZhZNEcq7bbFgyAQO7P", "segformer.b0.640x1280.city.160k.pth")
segformer_b0_cityscapes_768 = Builder("b0", 19, "1hMrg7e3z7iPHzb-jAKLwe5KD6jpzfXbY", "segformer.b0.768x768.city.160k.pth")
segformer_b0_cityscapes_1024 = Builder("b0", 19, "10lD5u0xVDJDKkIYxJDWkSeA2mfK_tgh9", "segformer.b0.1024x1024.city.160k.pth")
segformer_b1_ade20k_512 = Builder("b1", 150, "1PNaxIg3gAqtxrqTNsYPriR2c9j68umuj", "segformer.b1.512x512.ade.160k.pth")
segformer_b1_cityscapes_1024 = Builder("b1", 19, "1sSdiqRsRMhLJCfs0SydF7iKgeQNcXDZj", "segformer.b1.1024x1024.city.160k.pth")
segformer_b2_ade20k_512 = Builder("b2", 150, "13AMcdZYePbrTtwVzdJwZP5PF8PKehGhU", "segformer.b2.512x512.ade.160k.pth")
segformer_b2_cityscapes_1024 = Builder("b2", 19, "1MZhqvWDOKdo5rBPC2sL6kWL25JpxOg38", "segformer.b2.1024x1024.city.160k.pth")
segformer_b3_ade20k_512 = Builder("b3", 150, "16ILNDrZrQRJrXsIcSjUC56ueR72Rlant", "segformer.b3.512x512.ade.160k.pth")
segformer_b3_cityscapes_1024 = Builder("b3", 19, "1dc1YM2b3844-dLKq0qe77qb9_7brReIF", "segformer.b3.1024x1024.city.160k.pth")
segformer_b4_ade20k_512 = Builder("b4", 150, "171YHhri1rT5lwxmfPW76eU9DPP9OR27n", "segformer.b4.512x512.ade.160k.pth")
segformer_b4_cityscapes_1024 = Builder("b4", 19, "1F9QqGFzhr5wdX-FWax1xE2l7B8lqs42s", "segformer.b4.1024x1024.city.160k.pth")
segformer_b5_ade20k_640 = Builder("b5", 150, "11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz", "segformer.b5.640x640.ade.160k.pth")
segformer_b5_cityscapes_1024 = Builder("b5", 19, "1z3eFf-xVMkcb1Nmcibv6Ut-lTh81RLgO", "segformer.b5.1024x1024.city.160k.pth")
