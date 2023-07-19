import jax, fnx, inspect, einx
from functools import partial
import haiku as hk
import numpy as np
import jax.numpy as jnp
from typing import Union, Optional, Sequence, Tuple

Padding = Union[str, int]
Initializer = Union[hk.initializers.Initializer, float, int]

def generate_hk_padding(padding: Padding, kernel_size, dilation=None):
    if dilation is None:
        dilation = [1] * len(kernel_size)

    if isinstance(padding, str):
        string = padding.lower()
        if string == "same-asym":
            pad_fn = lambda effective_kernel_size: ((effective_kernel_size - 1) // 2, effective_kernel_size // 2)
        elif string == "same-sym":
            pad_fn = lambda effective_kernel_size: (effective_kernel_size // 2, effective_kernel_size // 2)
        elif string == "valid":
            pad_fn = lambda effective_kernel_size: (0, 0)
        else:
            raise ValueError(f"Invalid padding type {padding}")
    elif isinstance(padding, int):
        pad_fn = lambda effective_kernel_size: (padding, padding)
    else:
        raise ValueError("Invalid padding")

    effective_kernel_size = [(kernel_size - 1) * dilation + 1 for kernel_size, dilation in zip(kernel_size, dilation)]

    padding = [pad_fn(k) for k in effective_kernel_size]
    return tuple(padding)

def generate_hk_init(init: Initializer):
    if isinstance(init, int) or isinstance(init, float):
        return lambda shape, dtype: jnp.full(shape, init, dtype=dtype)
    else:
        return init

class ReshapedLinear(hk.Linear):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        shape = x.shape[:-1] + (self.output_size,)
        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        x = hk.Linear.__call__(self, x)
        x = x.reshape(shape)
        return x

@fnx.wrap_module
def conv(
    x: jnp.ndarray,
    bias: bool,
    weight_init: Initializer,
    bias_init: Initializer,
    padding: Padding,
    kernel_size: Union[int, Sequence[int]],
    channels: Optional[int] = None,
    stride: Union[int, Sequence[int]] = 1,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    name: str = "conv",
) -> jnp.ndarray:
    """Apply an n-dimensional convolution to the input ``x``.

    Args:
        x: Input tensor.
        bias: Whether to add bias or not.
        weight_init: Initializer of the weight matrix.
        bias_init: Initializer of the bias vector.
        padding:

            * If ``padding`` is an int, this will pad front and back with this amount.
            * If ``padding`` is ``"same-sym"``, this will pad such that the output has the same dimensions as the input with ``pad_front == pad_back``.
            * If ``padding`` is ``"same-asym"``, this will pad such that the output has the same dimensions as the input with ``pad_front <= pad_back``. Same as the Tensorflow behavior ``SAME``.
            * If ``padding`` is ``"valid"``, will pad with ``padding = 0``.

        kernel_size: Size of the convolution kernel. Either an integer or a sequence of n integers.
        channels: Number of output channels. Defaults to the number of channels of ``x``.
        stride: Stride of the convolution kernel. Either an integer or a sequence of n integers. Defaults to 1.
        dilation: Dilation rate of the convolution kernel. Either an integer or a sequence of n integers. Defaults to 1.
        groups: Number of feature groups of the convolution. If ``groups=="depthwise"``, will set ``groups`` to the number of channels of ``x``. Defaults to 1.
        name: Name of the module. Defaults to ``"conv"``.

    Returns:
        The convolution output.
    """

    if not isinstance(bias, bool):
        raise ValueError(f"Invalid bias value {bool}")
    if channels is None:
        channels = x.shape[-1]
    if isinstance(groups, str) and groups.lower() == "depthwise":
        groups = channels

    if len(x.shape) < 2:
        raise ValueError("Must have at least 2 dimensions")
    elif len(x.shape) > 2 or groups > 1:
        has_inserted_dim = len(x.shape) == 2
        if has_inserted_dim:
            assert stride == 1 and kernel_size == 1 and dilation == 1
            x = x[:, jnp.newaxis, :]

        num_spatial_dims = len(x.shape) - 2
        stride = fnx.util.replicate(stride, num_spatial_dims)
        kernel_size = fnx.util.replicate(kernel_size, num_spatial_dims)
        dilation = fnx.util.replicate(dilation, num_spatial_dims)

        if np.any(np.logical_and(kernel_size == 1, dilation != 1)):
            raise ValueError("dilation cannot be > 1 when kernel_size == 1")

        if np.all(np.asarray(kernel_size) == 1) and np.all(np.asarray(stride) == 1) and groups == 1:
            module = ReshapedLinear(
                output_size=channels,
                with_bias=bias,
                w_init=generate_hk_init(weight_init),
                b_init=generate_hk_init(bias_init),
                name=name,
            )
        else:
            module = hk.ConvND(
                num_spatial_dims=num_spatial_dims,
                output_channels=channels,
                kernel_shape=kernel_size,
                stride=stride,
                rate=dilation,
                with_bias=bias,
                padding=generate_hk_padding(padding, kernel_size, dilation),
                feature_group_count=groups,
                w_init=generate_hk_init(weight_init),
                b_init=generate_hk_init(bias_init),
                name=name,
            )
    else:
        if kernel_size != 1 or stride != 1:
            raise ValueError("kernel_size and stride must be 1 for two-dimensional inputs")
        module = hk.Linear(
            output_size=channels,
            with_bias=bias,
            w_init=generate_hk_init(weight_init),
            b_init=generate_hk_init(bias_init),
            name=name,
        )
    return module

# @wrap_module
# def conv_transpose(x, bias, weight_init, bias_init, padding, kernel_size, channels=None, stride=1, name="conv"):
#     if not isinstance(bias, bool):
#         raise ValueError(f"Invalid bias value {bool}")
#     if channels is None:
#         channels = x.shape[-1]
#
#     if len(x.shape) <= 2:
#         raise ValueError("Must have at least 3 dimensions")
#     elif len(x.shape) > 2:
#         num_spatial_dims = len(x.shape) - 2
#         stride = fnx.util.replicate(stride, num_spatial_dims, "stride")
#         kernel_size = fnx.util.replicate(kernel_size, num_spatial_dims, "kernel_size")
#
#         module = hk.ConvNDTranspose(
#             num_spatial_dims=num_spatial_dims,
#             output_channels=channels,
#             kernel_shape=kernel_size,
#             stride=stride,
#             rate=dilation,
#             with_bias=bias,
#             padding=generate_hk_padding(padding, kernel_size),
#             w_init=weight_init,
#             b_init=bias_init,
#             name=name,
#         )
#     return module

@fnx.module
def pool(
    x: jnp.ndarray,
    mode: str,
    padding: Padding,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    include_padded_values: bool = True,
) -> jnp.ndarray:
    """Apply n-dimensional spatial pooling to the input ``x``.

    Args:
        x: Input tensor.
        mode:

            * If ``mode`` is ``"max"``, applies maximum pooling.
            * If ``mode`` is ``"sum"``, applies sum pooling.
            * If ``mode`` is ``"avg"`` or ``"mean"``, applies average pooling.

        padding:

            * If ``padding`` is an int, this will pad front and back with this amount.
            * If ``padding`` is ``"same-sym"``, this will pad such that the output has the same dimensions as the input with ``pad_front == pad_back``.
            * If ``padding`` is ``"same-asym"``, this will pad such that the output has the same dimensions as the input with ``pad_front <= pad_back``. Default behavior in Tensorflow.
            * If ``padding`` is ``"valid"``, will pad with ``padding = 0``.

        kernel_size: Size of the pooling kernel. Either an integer or a sequence of n integers.
        stride: Stride of the pooling kernel. Either an integer or a sequence of n integers. Defaults to 1.
        include_padded_values: True if this should include the padded values in the pooling operation, False if the kernel_size should be reduced at the boundaries of ``x`` to include only valid values.
        name: Name of the module. Defaults to ``"pool"``.

    Returns:
        The pooling output.
    """

    num_spatial_dims = len(x.shape) - 2
    stride = fnx.util.replicate(stride, num_spatial_dims)
    kernel_size = fnx.util.replicate(kernel_size, num_spatial_dims)
    padding = generate_hk_padding(padding, kernel_size)

    kernel_size = (1,) + kernel_size + (1,)
    stride = (1,) + stride + (1,)
    padding = ((0, 0),) + padding + ((0, 0),)

    mode = mode.lower()
    if mode == "max":
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, kernel_size, stride, padding)
    elif mode == "sum":
        return jax.lax.reduce_window(x, 0, jax.lax.add, kernel_size, stride, padding)
    elif mode == "avg" or mode == "mean":
        pooled = jax.lax.reduce_window(x, 0, jax.lax.add, kernel_size, stride, padding)
        if include_padded_values or all(p[0] == 0 and p[1] == 0 for p in padding):
            return pooled / np.prod(kernel_size)
        else:
            denom = jax.lax.reduce_window(jnp.ones_like(x), 0, jax.lax.add, kernel_size, stride, padding)
            assert pooled.shape == denom.shape
            return pooled / denom
    else:
        raise ValueError(f"Invalid pooling mode {mode}")



def ema_on_train(f, decay_rate, name):
    ema = hk.ExponentialMovingAverage(decay_rate, name=name)

    if fnx.is_training:
        x = f()
        ema(f())
        return x
    else:
        return ema.average

@fnx.module(name="norm")
def meanvar_norm(
    x: jnp.ndarray,
    stats: str,
    params: str = "b... [c]",
    decay_rate: float = None,
    epsilon: float = 0,
    scale: bool = True,
    bias: bool = True,
    scale_init: Initializer = hk.initializers.Constant(1.0),
    bias_init: Initializer = hk.initializers.Constant(0.0),
) -> jnp.ndarray:
    """Normalize mean and variance of the input and add trainable scale and bias.

    Args:
        x: Input tensor.
        stats: einx expression for reduction operation that computes batch statistics.
        params: einx expression for element-wise operation that adds scale and bias. Defaults to "b... [c]".
        decay_rate: Decay rate of the running mean and variance. If None, this will only compute statistics for the current batch. Defaults to None.
        epsilon: Epsilon value to prevent division by zero. Defaults to 0.
        scale: Whether to add learnable scale parameter. Defaults to True.
        bias: Whether to add learnable bias parameter. Defaults to True.
        scale_init: Initializer of the scale vector. Defaults to ``hk.initializers.Constant(1.0)``.
        bias_init: Initializer of the bias vector. Defaults to ``hk.initializers.Constant(0.0)``.
        name: Name of the module. Defaults to ``"norm"``.

    Returns:
        The normalized output.

    Note:
        If decay_rate is not None, ``fnx.is_training`` needs to be set.
    """
    return einx.dl.meanvar_norm(
        x,
        stats=stats,
        params=params,
        moving_average=partial(ema_on_train, decay_rate=decay_rate) if not decay_rate is None else None,
        epsilon=epsilon,
        scale=partial(fnx.param, name="scale", init=generate_hk_init(scale_init), dtype=x.dtype) if scale else None,
        bias=partial(fnx.param, name="bias", init=generate_hk_init(bias_init), dtype=x.dtype) if bias else None,
    )

@fnx.module(name="norm")
def batchnorm(
    x: jnp.ndarray,
    decay_rate: float,
    epsilon: float,
    scale: bool = True,
    bias: bool = True,
    scale_init: Initializer = hk.initializers.Constant(1.0),
    bias_init: Initializer = hk.initializers.Constant(0.0),
) -> jnp.ndarray:
    """Apply Batch Normalization to the input.

    Args:
        x: Input tensor.
        decay_rate: Decay rate of the running mean and variance.
        epsilon: Epsilon value to prevent division by zero.
        scale: Whether to add learnable scale parameter. Defaults to True.
        bias: Whether to add learnable bias parameter. Defaults to True.
        scale_init: Initializer of the scale vector. Defaults to ``hk.initializers.Constant(1.0)``.
        bias_init: Initializer of the bias vector. Defaults to ``hk.initializers.Constant(0.0)``.
        name: Name of the module. Defaults to ``"norm"``.

    Returns:
        The normalized output.

    Note:
        Requires ``is_training`` to be set
    """
    return meanvar_norm(
        x,
        stats="[b...] c",
        params="b... [c]",
        decay_rate=decay_rate,
        epsilon=epsilon,
        scale=scale,
        bias=bias,
        scale_init=scale_init,
        bias_init=bias_init,
        skip_module=True,
    )

# See: https://github.com/apple/ml-cvnets/issues/34
@fnx.module(name="norm")
def layernorm(
    x: jnp.ndarray,
    epsilon: float,
    scale: bool = True,
    bias: bool = True,
    scale_init: Initializer = hk.initializers.Constant(1.0),
    bias_init: Initializer = hk.initializers.Constant(0.0),
) -> jnp.ndarray:
    """Apply Layer Normalization to the input.

    Args:
        x: Input tensor.
        epsilon: Epsilon value to prevent division by zero.
        scale: Whether to add learnable scale parameter. Defaults to True.
        bias: Whether to add learnable bias parameter. Defaults to True.
        scale_init: Initializer of the scale vector. Defaults to ``hk.initializers.Constant(1.0)``.
        bias_init: Initializer of the bias vector. Defaults to ``hk.initializers.Constant(0.0)``.
        name: Name of the module. Defaults to ``"norm"``.

    Returns:
        The normalized output.
    """
    return meanvar_norm(
        x,
        stats="b... [c]",
        params="b... [c]",
        epsilon=epsilon,
        scale=scale,
        bias=bias,
        scale_init=scale_init,
        bias_init=bias_init,
        skip_module=True,
    )

@fnx.module(name="norm")
def instancenorm(
    x: jnp.ndarray,
    epsilon: float,
    scale: bool = True,
    bias: bool = True,
    scale_init: Initializer = hk.initializers.Constant(1.0),
    bias_init: Initializer = hk.initializers.Constant(0.0),
) -> jnp.ndarray:
    """Apply Instance Normalization to the input.

    Args:
        x: Input tensor.
        epsilon: Epsilon value to prevent division by zero.
        scale: Whether to add learnable scale parameter. Defaults to True.
        bias: Whether to add learnable bias parameter. Defaults to True.
        scale_init: Initializer of the scale vector. Defaults to ``hk.initializers.Constant(1.0)``.
        bias_init: Initializer of the bias vector. Defaults to ``hk.initializers.Constant(0.0)``.
        name: Name of the module. Defaults to ``"norm"``.

    Returns:
        The normalized output.
    """
    return meanvar_norm(
        x,
        stats="b [s...] c",
        params="b... [c]",
        epsilon=epsilon,
        scale=scale,
        bias=bias,
        scale_init=scale_init,
        bias_init=bias_init,
        skip_module=True,
    )

@fnx.module(name="norm")
def groupnorm(
    x: jnp.ndarray,
    groups: int,
    epsilon: float,
    scale: bool = True,
    bias: bool = True,
    scale_init: Initializer = hk.initializers.Constant(1.0),
    bias_init: Initializer = hk.initializers.Constant(0.0),
) -> jnp.ndarray:
    """Apply Group Normalization to the input.

    Args:
        x: Input tensor.
        groups: Number of feature groups for which mean and variance are computed.
        epsilon: Epsilon value to prevent division by zero.
        scale: Whether to add learnable scale parameter. Defaults to True.
        bias: Whether to add learnable bias parameter. Defaults to True.
        scale_init: Initializer of the scale vector. Defaults to ``hk.initializers.Constant(1.0)``.
        bias_init: Initializer of the bias vector. Defaults to ``hk.initializers.Constant(0.0)``.
        name: Name of the module. Defaults to ``"norm"``.

    Returns:
        The normalized output.
    """
    return meanvar_norm(
        x,
        stats=("b [s...] (g [c])", {"g": groups}),
        params="b... [c]",
        epsilon=epsilon,
        scale=scale,
        bias=bias,
        scale_init=scale_init,
        bias_init=bias_init,
        skip_module=True,
    )

def conv_depthwise(*args, **kwargs) -> jnp.ndarray:
    """Apply an n-dimensional depthwise convolution to the input ``x``.

    Equivalent to ``fnx.conv`` with ``groups="depthwise"``.
    """
    return fnx.conv(*args, groups="depthwise", **kwargs)

def linear(*args, name="linear", **kwargs) -> jnp.ndarray:
    """Apply a linear layer to the input ``x``.

    Args:
        x: Input tensor.
        bias: Whether to add bias or not.
        weight_init: Initializer of the weight matrix.
        bias_init: Initializer of the bias vector.
        channels: Number of output channels. Defaults to the number of channels of ``x``.
        groups: Number of feature groups. If ``groups=="depthwise"``, will set ``groups`` to the number of channels of ``x``. Defaults to 1.
        name: Name of the module. Defaults to ``"linear"``.

    Returns:
        The output of the linear layer.
    """
    return fnx.conv(*args, stride=1, kernel_size=1, name=name, **kwargs)

@fnx.module
def scale(
    x: jnp.ndarray,
    expr: str,
    init: Initializer,
):
    """Apply an element-wise scale to the input ``x``.

    Args:
        x: Input tensor.
        expr: einx expression of the element-wise operation that multiplies x and scale.
        init: Initializer of the scale tensor.
        name: Name of the module. Defaults to ``"scale"``.

    Returns:
        The output of the scale layer.
    """
    return einx.multiply(expr, x, partial(fnx.param, name="scale", init=generate_hk_init(init)))

def layerscale(x, **kwargs):
    """Apply layer-scale to the input ``x``.

    Equivalent to ``fnx.scale`` with ``expr="b... [c]"``.

    Note:
        https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html
    """
    return fnx.scale(x, expr="b... [c]", **kwargs)
