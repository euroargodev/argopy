__author__ = 'sean.tokunaga@ifremer.fr'

import numpy as np


def get_param(param_name, param_adjusted_name, adj_mask, xarr, assign_nan=None):
    """
    from an xarray Dataset, extracts values as an np.ndarray for a parameter.
    :param param_name: Name of parameter to extract
    :param param_adjusted_name: Name of adjusted parameter to extract
    :param adj_mask: 1-D boolean array of length profile_n for deciding which param to choose from (adj or not)
    :param xarr: the xarray
    :param assign_nan: (default=None) if specified, nans, infs and -infs are replaced with this value
    :return: an np.ndarray of shape(param_name) (usually profile_n, level_n) with a mixture of param and param adjusted,
    according to adj_mask
    """
    vals = xarr[param_name].values  # as np.ndarray
    adj_vals = xarr[param_adjusted_name].values # as np.ndarray

    # detemrine which variable has the longer profile length.
    # (the fact that they're not always equal is a profoundly boring mystery)
    max_shape = np.max(np.array([vals.shape, adj_vals.shape]).transpose(), axis=-1)

    # Make an array full of nans with the right shape.
    merged_vals = np.empty(max_shape)
    merged_vals.fill(np.nan)
    merged_vals[adj_mask] = adj_vals[adj_mask]  # Insert values from adjusted column
    merged_vals[~adj_mask] = vals[~adj_mask]  # Insert values from non-adjusted column
    # Replace nans with some value of choice.
    if assign_nan is not None:
        merged_vals[~np.isfinite(merged_vals)] = assign_nan
    return merged_vals


def get_use_adj_map(adjusted_param_name, xarr):
    """
    Checks to see if the profile contains finite values.
    :param adjusted_param_name: Parameter to check
    :param xarr: xarray dataset
    :return: 1-D array of booleans: True if we should use PARAM_ADJUSTED else False.
    length of array == number of profiles.
    """
    return np.apply_along_axis(lambda x: np.isfinite(x).any(),
                               axis=1,
                               arr=xarr[adjusted_param_name].values.astype(np.float32))
    #return np.apply_along_axis(lambda x: np.logical_and(~np.isnan(x), (x != 9)).any(),
    #                           axis=1,
    #                           arr=xarr[adjusted_param_name].values.astype(np.float32))