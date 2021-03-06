from __future__ import annotations
import numpy as np
from math import sqrt


class Metrics():

    @staticmethod
    def mse_(y, y_hat):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching shape problem.
        Raises:
        This function should not raise any Exception.
        """
        try:
            mse = (1.0 / y.shape[0]) * np.sum((y - y_hat) ** 2, axis=0)
            return float(mse)
        except:
            return None

    @staticmethod
    def rmse_(y, y_hat):
        """
        Description:
        Calculate the RMSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
        rmse: has to be a float.
        None if there is a matching shape problem.
        Raises:
        This function should not raise any Exception.
        """
        try:
            rmse = sqrt(Metrics.mse_(y, y_hat))
            return float(rmse)
        except:
            return None

    @staticmethod
    def mae_(y, y_hat):
        """
        Description:
        Calculate the MAE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
        mae: has to be a float.
        None if there is a matching shape problem.
        Raises:
        This function should not raise any Exception.
        """
        try:
            mae = (1.0 / y.shape[0]) * np.sum(np.absolute(y - y_hat), axis=0)
            return float(mae)
        except:
            return None

    @staticmethod
    def r2score_(y, y_hat):
        """
        Description:
        Calculate the R2score between the predicted output and the output.
        Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
        r2score: has to be a float.
        None if there is a matching shape problem.
        Raises:
        This function should not raise any Exception.
        """
        try:
            mean = np.mean(y, axis=0)
            residual = np.sum((y_hat - y) ** 2, axis=0)
            m_var = np.sum((y - mean) ** 2, axis=0)
            r2 = 1 - (residual / m_var)
            return float(r2)
        except:
            return None


class MyLinearRegression(Metrics):
    """ Homemade linear regression class to fit like a tiny boss-ish
    """
    CLS_loss_fct = Metrics.mse_

    def __init__(self, thetas, alpha=1e-2, max_iter=1000):
        # Checking of the attributes:
        if (not isinstance(thetas, (np.ndarray, tuple, list))) \
            or (not isinstance(alpha, (int, float))) \
                or (not isinstance(max_iter, int)):
            s = "At least one of the parameters is not of expected type."
            raise TypeError(s)

        # Testing the shape of the paramters.
        thetas = self._convert_thetas_(thetas)
        if (alpha >= 1) or (alpha <= 0) or (max_iter <= 0):
            return None
        # Casting self.theta to float, in case it is integer
        self.thetas = thetas.astype('float64')
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def _convert_thetas_(thetas):
        if isinstance(thetas, np.ndarray):
            return thetas
        return np.array(thetas).reshape(-1, 1)

    def _gradient_(self, x, y):
        """ Private function gradient, there is no test perform on the
        parameters. It is to avoid to perform useless same tests as each
        call of gradient in fit method.
        """
        xp = np.hstack((np.ones((x.shape[0], 1)), x))
        return xp.T @ (xp @ self.thetas - y) / x.shape[0]

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array,
        without any for-loop. The three arrays must have compatible
        shapes.
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
                    or (not isinstance(self.thetas, np.ndarray)):
                return None

            # Testing the shape of the paramters.
            if (x.shape[1] != 1) or (y.shape[1] != 1) \
                or (self.thetas.shape[1] != 1) \
                    or (x.shape[0] != y.shape[0]):
                return None
            grad = self._gradient_(x, y)

            return grad
        except:
            return None

    def fit_(self, x, y):
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
            max_iter: has to be an int, the number of iterations done during
                      the gradient descent
        Return:
            new_theta: numpy.array, a vector of shape 2 * 1.
            None if there is a matching shape problem.
            None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking x, y and theta are numpy array
            if (not isinstance(x, np.ndarray)) \
                or (not isinstance(y, np.ndarray)) \
                    or (not isinstance(self.thetas, np.ndarray)):
                return None
            # Checking the shape of x, y and self.theta
            if (x.shape[1] != 1) \
                or (y.shape[1] != 1) \
                    or (x.shape[0] != y.shape[0]) \
                    or (self.thetas.shape[0] != x.shape[1] + 1):
                return None
            # Performing the gradient descent
            for _ in range(self.max_iter):
                grad = self._gradient_(x, y)
                self.thetas = self.thetas - self.alpha * grad
        except:
            # If something unexpected happened, we juste leave
            return None

    @staticmethod
    def loss_elem_(y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
                (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        try:
            res = (y - y_hat) ** 2
            return res
        except:
            None

    @staticmethod
    def loss_(y, y_hat):
        """Computes the half mean squared error of two non-empty numpy.array,
        without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            loss = MyLinearRegression.CLS_loss_fct(y, y_hat)
            # loss = (y - y_hat).T @ (y - y_hat) / (2.0 * y.shape[0])
            return float(loss) / 2.0
        except:
            None

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty
        numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a vector of shape 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of shape m * 1.
            None if x or theta are empty numpy.array.
            None if x or theta shapes are not appropriate.
            None if x or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(x, (np.ndarray)):
                return None
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if any([n == 0 for n in x.shape]):
                return None
            if self.thetas.shape != (x.shape[1] + 1, 1):
                return None
            xp = np.hstack((np.ones((x.shape[0], 1)), x))
            ypred = xp @ self.thetas
            return ypred
        except:
            return None
