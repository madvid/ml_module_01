import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using
    the min-max standardization.
    Args:
        x: has to be an numpy.array, a vector.
    Return:
        x’ as a numpy.array.
        None if x is a non-empty numpy.array or not a numpy.array.
        None if x is not of the expected type.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.ndim == 2:
        if x.shape[1] != 1:
            print("Note: You have passed an array with 2nd dim different"
                  + " than 1. Minmax score is calculated along axis 0.")
    try:
        x_cast = x.astype(np.float64)
        min = np.min(x_cast, axis=0)
        max = np.max(x_cast, axis=0)
        xp = (x_cast - min) / (max - min)
        return xp
    except:
        s = "Warning: Something wrong happened during " \
            + "calculation of the minmax-score."
        print(s)
        return None


if __name__ == "__main__":
    # Example 1:
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    print(minmax(X))
    # Output:
    expected_minmax = np.array([[0.58333333],
                                [1.],
                                [0.33333333],
                                [0.77777778],
                                [0.91666667],
                                [0.66666667],
                                [0.]])
    print("Expected minmax-score:\n", expected_minmax, "\n")

    # Example 2:
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print(minmax(Y))
    # Output:
    expected_minmax = np.array([[0.63636364],
                                [1.],
                                [0.18181818],
                                [0.72727273],
                                [0.93939394],
                                [0.6969697],
                                [0.]])
    print("\n\nExpected minmax-score:\n", expected_minmax)
