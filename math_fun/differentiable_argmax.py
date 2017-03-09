import numpy as np

def np_max_amplifier(x, amp=5, axis=None, transpose=False):
    """
    Amplifies to the greatest element of a tensor while reducing the others.

    The formula is: (x / max(x)) ^ amp

    Parameters
    ----------
    x: tf.Tensor
        The tensor to amplify. Should have numeric type.

    amp: int (optional)
        The power of the amplification.

    axis: (optional)
        The dimensions to reduce. If None (the default), reduces all dimensions.

    Returns
    -------
    The amplified tensor.

    """
    pos_x = np.add(x, np.abs(np.amin(x)))
    #pos_x = x
    amplified_x = np.power(np.divide(np.transpose(pos_x),
                                     np.add(np.amax(pos_x, axis=axis),
                                            0.1e-100))
                           , amp)

    if transpose:
        return amplified_x
    else:
        return np.transpose(amplified_x)


def np_differentiable_argmax(x, power=5, decoder=None, axis=None):
    if not decoder:
        decoder = np.array([[range(len(x))]], dtype=np.float)
    amplified_x = np_max_amplifier(x, power, axis=axis, transpose=True)
    return np.matmul(decoder, amplified_x)