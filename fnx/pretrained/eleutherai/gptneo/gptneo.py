import os, fnx, jax, tiktoken, functools
import haiku as hk
import jax.numpy as jnp
import numpy as np
from functools import partial

config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.gelu),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, variant, hf_name, heads):
        self.variant = variant
        self.hf_name = hf_name
        self.heads = heads

    encoder = property(functools.cache(lambda self: tiktoken.get_encoding("gpt2")))

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        # 1. Use local causal attention every second block
        # 2. No bias in qkv
        use_local = False
        def replace(next_interceptor, func, args, kwargs, context):
            nonlocal use_local
            if context.fullname.endswith("full_attention"):
                if use_local:
                    kwargs["mask"] = fnx.attention.mask.local_causal(args[0].shape[-2], window_size=256)
                use_local = not use_local
            elif context.fullname.endswith("qkv"):
                kwargs["bias"] = False

            return next_interceptor(func, args, kwargs)

        def model_fn(x):
            with config, fnx.intercept.custom(replace):
                x = vars(fnx.gpt)[f"gptneo_{self.variant}"](x)
                with fnx.scope("classifier"):
                    x = fnx.linear(x, channels=50257, bias=False, name="logits")
                    x = jax.nn.softmax(x, axis=-1)
            return x

        import transformers
        weights = {k: np.asarray(v) for k, v in transformers.AutoModelForCausalLM.from_pretrained(f"{self.hf_name}").state_dict().items()}

        # 3. Transpose weights
        transpose = lambda n, v: n.endswith("_proj.weight") or n.endswith("_fc.weight") or n.endswith("_head.weight")
        weights = {k: (np.transpose(v, (1, 0)) if transpose(k, v) else v) for k, v in weights.items()}
        # 4. They don't do variance-correction on attention logits
        i = 0
        while f"transformer.h.{i}.attn.attention.q_proj.weight" in weights:
            n = f"transformer.h.{i}.attn.attention.q_proj.weight"
            query_channels_per_head = weights[n].shape[-1] // self.heads
            weights[n] = weights[n] * (query_channels_per_head ** 0.5)
            i += 1
        # 5. They use separate layers for qkv, we use a single layer
        i = 0
        while f"transformer.h.{i}.attn.attention.k_proj.weight" in weights:
            query = f"transformer.h.{i}.attn.attention.q_proj.weight"
            key = f"transformer.h.{i}.attn.attention.k_proj.weight"
            value = f"transformer.h.{i}.attn.attention.v_proj.weight"
            weights[f"transformer.h.{i}.attn.attention.in_proj.weight"] = np.concatenate([weights[query], weights[key], weights[value]], axis=1)
            del weights[query]
            del weights[key]
            del weights[value]
            del weights[f"transformer.h.{i}.attn.attention.bias"]
            del weights[f"transformer.h.{i}.attn.attention.masked_bias"]
            i += 1

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("attn/norm", "ln_1")])

        return model_fn(x)

gptneo_125m_thepile = Builder("125m", "EleutherAI/gpt-neo-125M", 12)
gptneo_1_3b_thepile = Builder("1_3b", "EleutherAI/gpt-neo-1.3B", 16)
gptneo_2_7b_thepile = Builder("2_7b", "EleutherAI/gpt-neo-2.7B", 20)
