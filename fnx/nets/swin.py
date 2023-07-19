import fnx, einx
import jax.numpy as jnp
import numpy as np
import haiku as hk
from functools import partial

# Prior logits for every pair of pixels in a window
def prior_logits(window_size, heads):
    window_size = np.asarray(window_size)

    # Compute relative offset for every attention logit
    coords = jnp.stack(jnp.meshgrid(
        *[jnp.arange(s).astype("int32") for s in window_size],
        indexing="ij",
    ), axis=-1).reshape([-1, len(window_size)]) # (s...) i
    coords = coords[:, jnp.newaxis, :] - coords[jnp.newaxis, :, :]
    coords = coords + window_size[np.newaxis, np.newaxis, :] - 1 # (s1...) (s2...) i

    # Use the same prior logit for equal relative offsets
    prior_logits = fnx.param("prior_logits", shape=tuple(2 * window_size - 1) + (heads,), dtype="float32", init=hk.initializers.TruncatedNormal(0.02)) # (2w + 1)... h
    prior_logits = prior_logits[tuple(coords[..., i] for i in range(coords.shape[-1]))] # (s1...) (s2...) h
    prior_logits = einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", prior_logits)

    return prior_logits

# Mask on the attention logits such that each pixel interacts only with pixels in the same window
def mask(image_shape, shift, window_size):
    if np.any(shift > 0):
        # Create image where each pixel stores the id of its window
        window_ids = np.zeros(shape=image_shape, dtype="int32")
        id = 0
        def store_ids(slices):
            nonlocal id
            if len(slices) == len(window_size):
                window_ids[tuple(slices)] = id
                id += 1
            else:
                i = len(slices)
                for s in [slice(0, -window_size[i]), slice(-window_size[i], -shift[i]), slice(-shift[i], None)]:
                    store_ids(slices + [s])
        store_ids([])
        window_ids = einx.rearrange("(w s)... -> (w...) (s...)", window_ids, s=window_size)

        # Attention mask between two pixels is true if they have the same window id
        attn_mask = window_ids[:, :, jnp.newaxis] == window_ids[:, jnp.newaxis, :]
        attn_mask = einx.rearrange("(w...) (s1...) (s2...) -> b (w...) h (s1...) (s2...)", attn_mask, b=1, h=1)
        return attn_mask
    else:
        return None

@fnx.module
def block(x, shift, window_size, mlp_ratio=4, heads=1):
    shift = np.asarray(fnx.util.replicate(shift, len(x.shape) - 2))
    window_size = fnx.util.replicate(window_size, len(x.shape) - 2)

    with fnx.scope("attn"):
        x0 = x
        x = fnx.norm(x)

        # Shift and partition windows
        if np.any(shift > 0):
            x = jnp.roll(x, shift=-shift, axis=list(range(len(x.shape)))[1:-1])
        x = einx.rearrange("b (w s)... c -> b (w...) (s...) c", x, s=window_size)

        x = fnx.linear(x, channels=3 * x.shape[-1], bias=True, name="qkv")
        query, key, value = jnp.split(x, 3, axis=-1)
        x = fnx.attention.full_attention(query, key, value, heads=heads, prior_logits=prior_logits(window_size, heads), mask=mask(x0.shape[1:-1], shift, window_size))
        x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="out")

        # Unshift and merge windows
        x = einx.rearrange("b (w...) (s...) c -> b (w s)... c", x, s=window_size, output_shape=x0.shape)
        if np.any(shift > 0):
            x = jnp.roll(x, shift=shift, axis=list(range(len(x.shape)))[1:-1])

        x = x0 + fnx.stochastic.droppath(x)

    with fnx.scope("mlp"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=int(x0.shape[-1] * mlp_ratio), bias=True, name="linear1")
        x = fnx.act(x)
        x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="linear2")

        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def patch_embed(x, channels, patch_size):
    x = fnx.conv(x, channels=channels, kernel_size=patch_size, stride=patch_size, bias=True, padding=0) # TODO: bias should be false?
    x = fnx.norm(x)
    return x

@fnx.module
def downsample(x, channels, stride=2):
    stride = fnx.util.replicate(stride, len(x.shape) - 2)
    x = einx.rearrange("b (w s)... c -> b w... (s... c)", x, s=stride)

    x = fnx.norm(x)
    x = fnx.linear(x, channels=channels, bias=False)
    return x

@fnx.module
def encode(x, depths, channels, heads, patch_size, window_size):
    window_size = np.asarray(fnx.util.replicate(window_size, len(x.shape) - 2)) # TODO: should replicate return np.ndarray?
    x = patch_embed(x, channels=channels[0], patch_size=patch_size)

    for stage_index in range(len(depths)):
        if stage_index > 0:
            x = downsample(x, channels=channels[stage_index], name=f"downsample{stage_index + 1}")

        # Blocks
        with fnx.scope(f"stage{stage_index + 1}"):
            for block_index in range(depths[stage_index]):
                x = block(
                    x,
                    shift=0 if stage_index % 2 == 0 else window_size // 2,
                    window_size=window_size,
                    heads=heads[stage_index],
                    name=f"block{block_index + 1}",
                )
            fnx.sow("~", x)
    x = fnx.norm(x)

    return x

def swin_tiny(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 6, 2],
        channels=[96, 192, 384, 768],
        heads=[3, 6, 12, 24],
        patch_size=4,
        window_size=7,
        **kwargs,
    )

def swin_small(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 18, 2],
        channels=[96, 192, 384, 768],
        heads=[3, 6, 12, 24],
        patch_size=4,
        window_size=7,
        **kwargs,
    )

def swin_base(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 18, 2],
        channels=[128, 256, 512, 1024],
        heads=[4, 8, 16, 32],
        patch_size=4,
        window_size=7,
        **kwargs,
    )

def swin_large(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 18, 2],
        channels=[192, 384, 768, 1536],
        heads=[6, 12, 24, 48],
        patch_size=4,
        window_size=7,
        **kwargs,
    )
