import sys
import os
import numpy as np

## Collecting the path where gradient method is
path = os.path.join(os.path.dirname(__file__), '..', 'ex01')
sys.path.insert(1, path)
from vec_gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1:
           (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1:
           (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done
                  during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected
             type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        ## Checking x, y and theta are numpy array
        if (not isinstance(x, np.ndarray)) \
            or (not isinstance(y, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)):
            return None
        ## Checking the shape of x, y and theta
        if (x.shape[1] != 1) \
            or (y.shape[1] != 1) \
                or (x.shape[0] != y.shape[0]) \
                or (theta.shape[0] != x.shape[1] + 1):
            return None
        ## Checking the type and values of max_iter and alpha
        if (not isinstance(max_iter, int)) \
            or (max_iter < 0) \
                or (not isinstance(alpha, float)) \
                or (alpha >= 1 or alpha <= 0):
            return None
        ## Casting theta to float, in case it is integer
        new_theta = np.copy(theta.astype('float64'))
        for _ in range(max_iter):
            grad = gradient(x, y, new_theta)
            new_theta[0] = new_theta[0] - alpha * grad[0]
            new_theta[1] = new_theta[1] - alpha * grad[1]
        return new_theta
    except:
        ## If something unexpected happened, we juste leave
        return None
