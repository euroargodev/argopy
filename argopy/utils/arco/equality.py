__author__ = 'sean.tokunaga@ifremer.fr'

import numpy as np

"""
A few helper functions for testing equality.
"""

def element_wise_nan_equal(a, b):
    """
    np.nan == np.nan returns False. I don't think we want this in our case. So I wrote an equality test where it
    returns True for all equal elements including nans
    :param a: numpy array
    :param b: numpy array
    :return:
    """
    return (a == b) + np.logical_and(np.isnan(a), np.isnan(b))


def nan_equal(a, b):
    """
    This is equivalent to element_wise_nan_equal(a, b).all()
    :param a: numpy array
    :param b: numpy array
    :return:
    """
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True
