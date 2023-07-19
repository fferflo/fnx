import os, fnx, jax, tiktoken, functools
import haiku as hk
import jax.numpy as jnp
import numpy as np

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, variant, hf_name):
        self.variant = variant
        self.hf_name = hf_name

    encoder = property(functools.cache(lambda self: tiktoken.get_encoding("gpt2")))

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with config:
                x = vars(fnx.gpt)[f"gpt2_{self.variant}"](x)
                with fnx.scope("classifier"):
                    x = fnx.linear(x, channels=50257, bias=False, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        import transformers
        weights = {k: np.asarray(v) for k, v in transformers.GPT2LMHeadModel.from_pretrained(f"{self.hf_name}").state_dict().items()}
        weights["lm_head.weight"] = np.transpose(weights["lm_head.weight"], (1, 0))
        weights = {k: v for k, v in weights.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}
        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "ln_1")])

        return model_fn(x)

gpt2_small = Builder("small", "gpt2")
gpt2_medium = Builder("medium", "gpt2-medium")
gpt2_large = Builder("large", "gpt2-large")
gpt2_xlarge = Builder("xlarge", "gpt2-xl")
