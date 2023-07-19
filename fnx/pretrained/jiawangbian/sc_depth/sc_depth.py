# Source: https://github.com/JiawangBian/sc_depth_pl
import os, fnx, jax, re
import haiku as hk
from functools import partial
import jax.numpy as jnp
import numpy as np

color_mean = np.asarray([0.45, 0.45, 0.45])
color_std = np.asarray([0.225, 0.225, 0.225])

def preprocess(color):
    color = (color / 255.0 - color_mean) / color_std
    return color

encoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, fnx.module(jax.nn.relu, name="act")),
    fnx.config.pytorch,
)

decoder_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.act, jax.nn.elu),
    fnx.config.pytorch,
)



class Builder:
    def __init__(self, name, onedrive):
        self.name = name
        self.onedrive = onedrive
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            input_shape = x.shape
            with encoder_config:
                xs = fnx.resnet.resnet_v1_18(x, stem=fnx.resnet.stem_b, reap=["stem/act", "stage1", "stage2", "stage3", "stage4"])
            with decoder_config:
                x = fnx.sc_depth.decode(xs, channels=16)

                with fnx.scope("to_depth"):
                    alpha = 10.0
                    beta = 0.01
                    x = jnp.pad(x, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="reflect")
                    x = fnx.conv(x, channels=1, kernel_size=3, stride=1, bias=True, padding=0)[..., 0]
                    x = alpha * jax.nn.sigmoid(x) + beta
                    x = 1.0 / x
                    x = jax.image.resize(x, shape=input_shape[:3], method="bilinear")
            return x

        file = fnx.pretrained.weights.download_onedrive(self.name, self.onedrive)
        weights = fnx.pretrained.weights.load_pytorch(file)
        weights = {k: v for k, v in weights.items() if k.startswith("depth_net") and not k.startswith("depth_net.encoder.encoder.fc.") and (not k.startswith("depth_net.decoder.dispconvs.") or k.startswith("depth_net.decoder.dispconvs.0"))}
        model_fn = fnx.pretrained.weights.init(model_fn, weights)

        return model_fn(x)

sc_depth_v3_kitti = Builder("sc_depth_v3_kitti.ckpt", "https://onedrive.live.com/?authkey=!AN9KaLjLL78kdKY&cid=36712431A95E7A25&id=36712431A95E7A25!4269&parId=36712431A95E7A25!4260")
sc_depth_v3_tum = Builder("sc_depth_v3_kitti.ckpt", "https://onedrive.live.com/?authkey=!AN9KaLjLL78kdKY&cid=36712431A95E7A25&id=36712431A95E7A25!4273&parId=36712431A95E7A25!4265")
sc_depth_v3_nyu = Builder("sc_depth_v3_kitti.ckpt", "https://onedrive.live.com/?authkey=!AN9KaLjLL78kdKY&cid=36712431A95E7A25&id=36712431A95E7A25!4271&parId=36712431A95E7A25!4264")
sc_depth_v3_ddad = Builder("sc_depth_v3_kitti.ckpt", "https://onedrive.live.com/?authkey=!AN9KaLjLL78kdKY&cid=36712431A95E7A25&id=36712431A95E7A25!4267&parId=36712431A95E7A25!4258")
sc_depth_v3_bonn = Builder("sc_depth_v3_kitti.ckpt", "https://onedrive.live.com/?authkey=!AN9KaLjLL78kdKY&cid=36712431A95E7A25&id=36712431A95E7A25!4266&parId=36712431A95E7A25!4257")
