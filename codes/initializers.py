import numpy as np


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
      shape: Integer shape tuple
    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


def glorot_uniform(shape, dtype=np.float64, seed=None):
    fan_in, fan_out = _compute_fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=shape).astype(dtype)
