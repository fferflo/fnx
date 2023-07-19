import pyunpack, os, fnx, jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import haiku as hk

def preprocess(color):
    color = color - np.array([123.68, 116.779, 103.939])
    color = color[..., ::-1]
    return color

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.relu),
    fnx.config.tensorflow,
)

class Builder:
    def __init__(self, variant, url):
        self.encoder = vars(fnx.resnet)[f"resnet_v1_{variant}"]
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

        file_compressed = fnx.pretrained.weights.download(self.url)
        file = file_compressed[:-len("_2016_08_28.tar.gz")] + ".ckpt"
        if not os.path.isfile(file):
            pyunpack.Archive(file_compressed).extractall(os.path.dirname(file_compressed))

        weights = fnx.pretrained.weights.load_tensorflow(file)
        weights = {k: v for k, v in weights.items() if not ("global_step" in k or "mean_rgb" in k)}
        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("/shortcut/norm/", "/shortcut/BatchNorm/")])

        return model_fn(x)

resnet_v1b_50_imagenet = Builder(50, "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz")
resnet_v1b_101_imagenet = Builder(101, "http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz")
resnet_v1b_152_imagenet = Builder(152, "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz")
