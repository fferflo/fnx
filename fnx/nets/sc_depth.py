import fnx, jax
import jax.numpy as jnp

@fnx.module
def decode(xs, channels):
    channels = [channels * (2 ** i) for i in range(len(xs))]

    x = xs[-1]
    for i in reversed(range(len(xs))):
        with fnx.scope(f"stage{i + 1}"):
            x = jnp.pad(x, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="reflect") # TODO: add padding_mode="reflect" to fnx.conv
            x = fnx.conv(x, channels=channels[i], kernel_size=3, stride=1, bias=True, padding=0, name="conv1")
            x = fnx.act(x)

            if i > 0:
                x = jnp.concatenate([
                    jax.image.resize(x, shape=(x.shape[0], xs[i - 1].shape[1], xs[i - 1].shape[2], x.shape[3]), method="bilinear"),
                    xs[i - 1],
                ], axis=-1)

            x = jnp.pad(x, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="reflect")
            x = fnx.conv(x, channels=channels[i], kernel_size=3, stride=1, bias=True, padding=0, name="conv2")
            x = fnx.act(x)
    return x
