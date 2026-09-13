"""
Function to map or extract information out of Argo xarray objects

Parameters
----------
xarray objects or list of xarray objects

Returns
-------
xarray objects or list of xarray objects
"""
from typing import Any
import xarray as xr


def map_vars_to_dict(ds: xr.Dataset, var_key: str, var_val:str, duplicate:bool=False) -> dict[Any, Any]:
    """Make a dictionary mapping 2 variables sharing one dimension

    This function can be used if in an Argo dataset, two variables share a similar dimension, and one want to map one variable values onto the other's (1:1 relation is expected, but many:1 can be handled as well).

    Typically:
     - `PARAM_NAME[N_PARAM]` and `PARAM_VALUE[N_PARAM]` share `N_PARAM`,
     - create a dictionary with `PARAM_NAME` as keys and `PARAM_VALUE` as values.

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        The dataset to work with
    var_key: str
        Name of the :class:`xarray.Dataset` variable to use as **dictionary keys**. Must have a single dimension shared with ``var_val``. String types are automatically stripped of white spaces.
    var_val: str
        Name of the :class:`xarray.Dataset` variable to use as **dictionary values**. Must have a single dimension shared with ``var_key``. String types are automatically stripped of white spaces.
    duplicate: bool, optional, default = False
        Should raise an error if `var_key` has duplicate values. If set to True, the last occurrence is used.

    Returns
    -------
    :class:`xarray.Dataset`

    Examples
    --------
    .. code-block::python

        # Load some data
        from argopy import ArgoFloat
        ds = ArgoFloat(6903091).open_dataset('meta')

        # Use the mapper:
        map_vars_to_dict(ds, 'LAUNCH_CONFIG_PARAMETER_NAME', 'LAUNCH_CONFIG_PARAMETER_VALUE')
        map_vars_to_dict(ds, 'SENSOR', 'SENSOR_MODEL')

    """
    assert ds[var_key].dims == ds[var_val].dims, f"{var_key} and {var_val} must have similar dimensions"
    assert len(ds[var_key].dims) == 1, f"{var_key} and {var_val} must have a single dimension"
    read = lambda ds, key: ds[key].item().strip() if ds[key].dtype.kind == 'U' else ds[key].item()

    shared_dim = ds[var_key].dims[0]
    result = {}
    for ii in ds[shared_dim]:
        this_param = ds.sel({shared_dim: ii})
        pname = read(this_param, var_key)
        pvalue = read(this_param, var_val)
        if pname in result and not duplicate:
            raise ValueError(f"{var_key}={pname} has more than one occurrence. Use the option 'duplicate=True' to keep the last occurrence.")
        result.update({pname: pvalue})
    return result
