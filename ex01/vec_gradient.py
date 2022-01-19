import sys
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from prediction import predict_


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Testing the type of the parameters, numpy array expected.
        if (not isinstance(x, np.ndarray)) \
            or (not isinstance(y, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)):
            return None

        # Testing the shape of the paramters.
        if (x.shape[1] != 1) \
            or (y.shape[1] != 1) \
                or (theta.shape[1] != 1) \
                or (x.shape[0] != y.shape[0]):
            return None
        m, grad = x.shape[0], np.zeros(theta.shape)

        xp = np.hstack((np.ones((m, 1)), x))
        grad = xp.T @ (xp @ theta - y)

        return grad / m
    except:
        return None
