import os, fnx, jax
import haiku as hk
from functools import partial
import jax.numpy as jnp

from ..common import color_mean, color_std, preprocess, config

class Builder:
    def __init__(self, variant, url):
        self.encoder = vars(fnx.convnext)[f"convnext_{variant}"]
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                x = self.encoder(x, block=fnx.convnext.block_v2)
                with fnx.scope("classifier"):
                    x = jnp.mean(x, axis=(1, 2))
                    x = fnx.norm(x)
                    x = fnx.linear(x, channels=1000, bias=True, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        model_fn = fnx.pretrained.weights.init(model_fn, weights)

        return model_fn(x)

convnext_v2_atto_ss_224 = Builder("atto", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt")
convnext_v2_femto_ss_224 = Builder("femto", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt")
convnext_v2_pico_ss_224 = Builder("pico", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt")
convnext_v2_nano_ss_224 = Builder("nano", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt")
convnext_v2_tiny_ss_224 = Builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt")
convnext_v2_base_ss_224 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt")
convnext_v2_large_ss_224 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt")
convnext_v2_huge_ss_224 = Builder("huge", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt")

convnext_v2_atto_imagenet1k_224 = Builder("atto", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt")
convnext_v2_femto_imagenet1k_224 = Builder("femto", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt")
convnext_v2_pico_imagenet1k_224 = Builder("pico", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt")
convnext_v2_nano_imagenet1k_224 = Builder("nano", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt")
convnext_v2_tiny_imagenet1k_224 = Builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt")
convnext_v2_base_imagenet1k_224 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt")
convnext_v2_large_imagenet1k_224 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt")
convnext_v2_huge_imagenet1k_224 = Builder("huge", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt")

convnext_v2_nano_imagenet22k_imagenet1k_224 = Builder("nano", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt")
convnext_v2_tiny_imagenet22k_imagenet1k_224 = Builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt")
convnext_v2_base_imagenet22k_imagenet1k_224 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt")
convnext_v2_large_imagenet22k_imagenet1k_224 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt")

convnext_v2_nano_imagenet22k_imagenet1k_384 = Builder("nano", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt")
convnext_v2_tiny_imagenet22k_imagenet1k_384 = Builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt")
convnext_v2_base_imagenet22k_imagenet1k_384 = Builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt")
convnext_v2_large_imagenet22k_imagenet1k_384 = Builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt")
convnext_v2_huge_imagenet22k_imagenet1k_384 = Builder("huge", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt")

convnext_v2_huge_imagenet22k_imagenet1k_512 = Builder("huge", f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt")
