import jax, fnx, einx
import jax.numpy as jnp
import haiku as hk

@fnx.module
def nonnegative_matrix_factorization(x, bases_num=64, epsilon=1e-6, iterations=6):
    def init(shape, dtype):
        b = jax.random.uniform(hk.next_rng_key(), minval=0.0, maxval=1.0, shape=shape)
        b = b.astype(dtype)
        b = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        return b
    bases = fnx.state(
        "bases",
        shape=[1, bases_num, x.shape[-1]],
        dtype=x.dtype,
        init=init,
    )
    bases = einx.rearrange("1 r c -> b r c", bases, b=x.shape[0])

    x_nograd = jax.lax.stop_gradient(x)
    bases_nograd = jax.lax.stop_gradient(bases)
    coef = einx.dot("b s... c, b r c -> b s... r", x_nograd, bases_nograd)
    coef = jax.nn.softmax(100.0 * coef, axis=-1)
    for _ in range(iterations):
        num = einx.dot("b s... c, b r c -> b s... r", x_nograd, bases_nograd)
        bases_2 = einx.dot("b r1 c, b r2 c -> b r1 r2", bases_nograd, bases_nograd)
        denom = einx.dot("b s... r1, b r1 r2 -> b s... r2", coef, bases_2)
        coef = coef * num / (denom + epsilon)

        num = einx.dot("b s... c, b s... r -> b r c", x_nograd, coef)
        coef_2 = einx.dot("b s... r1, b s... r2 -> b r1 r2", coef, coef)
        denom = einx.dot("b r1 c, b r1 r2 -> b r2 c", bases_nograd, coef_2)
        bases_nograd = bases_nograd * num / (denom + epsilon)

    num = einx.dot("b s... c, b r c -> b s... r", x, bases)
    bases_2 = einx.dot("b r1 c, b r2 c -> b r1 r2", bases, bases)
    denom = einx.dot("b s... r1, b r1 r2 -> b s... r2", coef, bases_2)
    coef = coef * num / (denom + epsilon)

    x = einx.dot("b r c, b s... r -> b s... c", bases, coef)

    return x

@fnx.module
def decode(xs, channels, matrix_decomposition=nonnegative_matrix_factorization):
    xs = [x for x in xs]

    for i in range(1, len(xs)):
        xs[i] = jax.image.resize(xs[i], xs[0].shape[:-1] + (xs[i].shape[-1],), method="bilinear")
    x = jnp.concatenate(xs, axis=-1)

    with fnx.scope("fuse"):
        x = fnx.linear(x, channels=channels, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    with fnx.scope("ham"):
        x0 = x

        with fnx.scope("in"):
            x = fnx.linear(x, bias=True)
            x = fnx.act(x)

        x = matrix_decomposition(x)

        with fnx.scope("out"):
            x = fnx.linear(x, bias=False)
            x = fnx.norm(x)
            x = x + x0
            x = fnx.act(x)

    with fnx.scope("align"):
        x = fnx.linear(x, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    return x
