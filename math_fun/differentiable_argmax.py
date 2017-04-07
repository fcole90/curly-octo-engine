import numpy as np

def np_max_amplifier(x, amp=5, axis=None, transpose=False, zero_break=0.1e-100):
    """
    Amplifies the greatest element of a tensor while reducing the others.

    The formula is: (x / max(x)) ^ amp.

    In addition, x values are shifted forward by the min value to allow
    all values to be non negative.

    To overcome zero-division errors, a small zero_break value is added to the
    denominator.

    Parameters
    ----------
    x: np.array
        The tensor to amplify. Should have numeric type.

    amp: int (optional)
        The power of the amplification.

    axis: (optional)
        The dimensions to reduce. If None (the default), reduces all dimensions.

    transpose: bool (optional)
        Flag to keep x transposed. False by default.

    zero_break: float (optional)
        Small amount to be added to the denominator of the division to avoid
        zero-division errors. Defaults to 0.1e-100. Always set this value to
        a number some order of magnitude below the max of x.

    Returns
    -------
    The amplified tensor.

    """
    pos_x = np.add(x, np.abs(np.amin(x)))
    #pos_x = x
    amplified_x = np.power(np.divide(np.transpose(pos_x),
                                     np.add(np.amax(pos_x, axis=axis),
                                            zero_break))
                           , amp)

    if transpose:
        return amplified_x
    else:
        return np.transpose(amplified_x)


def np_differentiable_argmax(x, power=5, decoder=None, axis=None):
    """
    Differentiable variant of np.argmax.

    Should give the same results as np.argmax, but this is differentiable and can
    be used as cost function of the gradient descent.

    Parameters
    ----------
    x: np.array
        An input Tensor.

    power: int (optional)
        The highest the power, the highest the precision.

    decoder: list (optional)
        An optional list of numerical values. Using a custom decoder, instead
        of the max argument, will be returned the value which has the index
        corresponding to the argmax: decoder[argmax(x)].

    axis:
        A Tensor. Must be one of the following types: int32, int64. int32,
        0 <= axis < rank(input). Describes which axis of the input Tensor
        to reduce across. For vectors, use axis = 0.

    Returns
    -------
    float
        The index of the greatest argument along the specified axis.
    """
    if not decoder:
        decoder = np.array([[range(x.shape[1])]], dtype=np.float)
    amplified_x = np_max_amplifier(x, power, axis=axis, transpose=True)
    return np.matmul(decoder, amplified_x)