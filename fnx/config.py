import fnx
import haiku as hk

@fnx.module
def norm(*args, **kwargs):
    """Dummy function that should be replaced using fnx.intercept.

    Raises:
        NotImplementedError: If called directly.
    """
    raise NotImplementedError("This is a function dummy that should be replaced via fnx.intercept")

@fnx.module
def act(*args, **kwargs):
    """Dummy function that should be replaced using fnx.intercept.

    Raises:
        NotImplementedError: If called directly.
    """
    raise NotImplementedError("This is a function dummy that should be replaced via fnx.intercept")

tensorflow = fnx.intercept.chain(
    fnx.intercept.defaults(fnx.batchnorm, epsilon=1e-3, decay_rate=0.99),
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-3),
    fnx.intercept.defaults(fnx.instancenorm, epsilon=1e-3),
    fnx.intercept.defaults(fnx.groupnorm, epsilon=1e-3),
    fnx.intercept.defaults(fnx.conv, padding="same-asym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"), bias_init=hk.initializers.Constant(0.0)),
    fnx.intercept.defaults(fnx.pool, padding="same-asym", include_padded_values=True),
)
"""
Context manager that uses ``fnx.intercept`` to switch default behavior to Tensorflow.
"""

pytorch = fnx.intercept.chain(
    fnx.intercept.defaults(fnx.batchnorm, epsilon=1e-5, decay_rate=0.9),
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.instancenorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.groupnorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.conv, padding="same-sym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"), bias_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")),
    fnx.intercept.defaults(fnx.pool, padding="same-sym", include_padded_values=True),
)
"""
Context manager that uses ``fnx.intercept`` to switch default behavior to PyTorch.
"""

haiku = fnx.intercept.chain(
    fnx.intercept.defaults(fnx.batchnorm, epsilon=1e-5, decay_rate=0.99),
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.instancenorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.groupnorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.conv, padding="same-asym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"), bias_init=hk.initializers.Constant(0.0)),
    # fnx.intercept.defaults(fnx.conv_transpose, padding="same-asym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"), bias_init=hk.initializers.Constant(0.0)),
    fnx.intercept.defaults(fnx.pool, padding="same-asym", include_padded_values=False),
)
"""
Context manager that uses ``fnx.intercept`` to switch default behavior to Haiku.
"""

flax = fnx.intercept.chain(
    fnx.intercept.defaults(fnx.batchnorm, epsilon=1e-5, decay_rate=0.99),
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-6),
    fnx.intercept.defaults(fnx.instancenorm, epsilon=1e-6),
    fnx.intercept.defaults(fnx.groupnorm, epsilon=1e-6),
    fnx.intercept.defaults(fnx.conv, padding="same-asym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"), bias_init=hk.initializers.Constant(0.0)),
    fnx.intercept.defaults(fnx.pool, padding="same-asym", include_padded_values=False),
)
"""
Context manager that uses ``fnx.intercept`` to switch default behavior to Flax.
"""

jittor = fnx.intercept.chain(
    fnx.intercept.defaults(fnx.batchnorm, epsilon=1e-5, decay_rate=0.9),
    fnx.intercept.defaults(fnx.layernorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.instancenorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.groupnorm, epsilon=1e-5),
    fnx.intercept.defaults(fnx.conv, padding="same-sym", weight_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"), bias_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")),
    fnx.intercept.defaults(fnx.pool, padding="same-sym", include_padded_values=True),
)
"""
Context manager that uses ``fnx.intercept`` to switch default behavior to Jittor.
"""
