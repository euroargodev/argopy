"""
Mathematically or statistically compute something out of xarray objects
"""

import numpy as np
from scipy import interpolate
import xarray as xr
from packaging import version
import logging

try:
    import gsw

    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False


from ..errors import InvalidOption

log = logging.getLogger("argopy.utils.compute")

#
#  From xarrayutils : https://github.com/jbusecke/xarrayutils/blob/master/xarrayutils/vertical_coordinates.py
#  Direct integration of those 2 functions to minimize dependencies and possibility of tuning them to our needs
#


def linear_interpolation_remap(
    z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped"
    # interpolation called in xarray ufunc
    def _regular_interp(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]

        # Need at least 5 points in the profile to interpolate, otherwise, return NaNs
        if len(y) < 5:
            interpolated = np.empty(len(target_values))
            interpolated[:] = np.nan
        else:
            # replace nans in target_values without of bound Values (just in case)
            target_values = np.where(
                ~np.isnan(target_values), target_values, np.nanmax(x) + 1
            )
            # Interpolate with fill value parameter to extend min pressure toward 0
            interpolated = interpolate.interp1d(
                x, y, bounds_error=False, fill_value=(y[0], y[-1])
            )(target_values)
        return interpolated

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a dataset input just apply the function for each appropriate array

    if version.parse(xr.__version__) > version.parse("0.15.0"):
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
            },
        )
    else:
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
        )
    remapped = xr.apply_ufunc(_regular_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped


def pchip_interpolation_remap(
    z,
    data,
    z_regridded,
    z_dim=None,
    z_regridded_dim="regridded",
    output_dim="remapped",
    zTolerance=None,
):
    """Vertical interpolation of one dataset variable, using Piecewise Cubic Hermite Interpolating Polynomial

    Parameters
    ----------
    z: :class:`xarray.DataArray`
        Input vertical pressure axis (eg: ds['PRES'])
    data: :class:`xarray.DataArray`
        Input variable to interpolate (eg: ds['TEMP'])
    z_regridded:
        Output vertical pressure axis (eg: ds['Z_LEVELS'])
    z_dim: default = None
        Input vertical dimension name (eg: "N_LEVELS"). Inferred from ``z`` by default.
    z_regridded_dim:
        Eg: "Z_LEVELS"
    output_dim:
    """
    if not with_gsw:
        raise ValueError("The pchip interpolation method requires the gsw librairy")

    def _regular_interp(x, y, xi, xTolerance=None):
        """Interpolation method called by xarray ufunc

        This consumes one single profile data as low-level structures, like numpy 1D arrays

        Follows Barker and McDougall (2020) requirements
        """
        yi_empty = np.full(
            len(xi), np.nan, dtype=np.float32
        )  # Output when something fails

        # 'mask' holds array index to keep for interpolation

        # Un-select nans from input x and y:
        mask = np.logical_and(~np.isnan(x), ~np.isnan(y))

        # Check for Barker and McDougall (2020) requirements:
        # Un-select negative pressures:
        mask = np.logical_and(mask, x > 0)

        # At least 5 points:
        if np.nonzero(mask)[0].shape[0] < 5:
            log.debug("Not enough points to work with, skip profile interpolation")
            return yi_empty

        # Skip a profile without monotonically increasing Pressure
        # todo: add test to remove profile with a density inversions > 0.03kg/m3
        dx = np.diff(x)
        if np.any(dx < 0):
            log.debug(
                "Encounter profile with unstable pressure, skip profile interpolation"
            )
            return yi_empty

        # Apply mask:
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            log.debug(
                "No data left to interpolate after pre-processing, skip profile interpolation"
            )
            return yi_empty

        # Run interp:
        yi = gsw.pchip_interp(x, y, xi, axis=0)

        # Follow EasyOneArgo - Lite specs for clean-up:

        # "Clean up interpolated points outside of the pressure range of the input profile end points"
        yi[np.argwhere(xi < np.nanmin(x))] = np.nan
        yi[np.argwhere(xi > np.nanmax(x))] = np.nan

        # "Clean up interpolated points where the pressure gap in the input data is greater than Ptolerance.
        # Ptolerance is set to vary with pressure, with larger values at deeper pressures because T/S
        # gradients are smaller at deeper pressures.
        # Ptolerance is set as 3 x local vertical spacing by default by ds.argo.interp_std_levels
        # This means for every two input points, at most 3 interpolated points are allowed."

        # Interpolate pressure tolerance values to input pressure levels
        p_tolerance_lookup = np.interp(x[:-1], xi, xTolerance)

        # Loop through each pressure level:
        for id_lev in range(len(x) - 1):
            # Toss out points where input pressure gap is greater than tolerance
            if dx[id_lev] > p_tolerance_lookup[id_lev]:
                id_del = np.where((xi > x[id_lev]) & (xi < x[id_lev + 1]))[0]
                yi[id_del] = np.nan

        return yi

    # Infer dimension from input:
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # Set kwargs for xarray.apply_ufunc miscellaneaous versions:
    if zTolerance is None:
        input_core_dims = [[dim], [dim], [z_regridded_dim]]
    else:
        input_core_dims = [[dim], [dim], [z_regridded_dim], [z_regridded_dim]]

    if version.parse(xr.__version__) > version.parse("0.15.0"):
        kwargs = dict(
            input_core_dims=input_core_dims,
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
            },
        )
    else:
        kwargs = dict(
            input_core_dims=input_core_dims,
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
        )
    if zTolerance is None:
        remapped = xr.apply_ufunc(_regular_interp, z, data, z_regridded, **kwargs)
    else:
        remapped = xr.apply_ufunc(
            _regular_interp, z, data, z_regridded, zTolerance, **kwargs
        )

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped


def mrst_pchip_interpolatation(
    z,
    Tdata,
    Sdata,
    z_regridded,
    z_dim=None,
    z_regridded_dim="regridded",
    output_dim="remapped",
    zTolerance=None,
):
    """Vertical interpolation of temperature/salinity, using the MRST-PCHIP method

    MRST-PCHIP: Multiply-Rotated Salinity-Temperature PCHIP Method

    """
    pass



def groupby_remap(
    z,
    data,
    z_regridded,  # noqa C901
    z_dim=None,
    z_regridded_dim="regridded",
    output_dim="remapped",
    select="deep",
    right=False,
):
    """todo: Need a docstring here !"""

    # sub-sampling called in xarray ufunc
    def _subsample_bins(x, y, target_values):
        # remove all nans from input x and y
        try:
            idx = np.logical_or(np.isnan(x), np.isnan(y))
        except TypeError:
            log.debug(
                "Error with this '%s' y data content: %s" % (type(y), str(np.unique(y)))
            )
            raise
        x = x[~idx]
        y = y[~idx]

        ifound = np.digitize(
            x, target_values, right=right
        )  # ``bins[i-1] <= x < bins[i]``
        ifound -= 1  # Because digitize returns a 1-based indexing, we need to remove 1
        y_binned = np.ones_like(target_values) * np.nan

        for ib, this_ibin in enumerate(np.unique(ifound)):
            ix = np.where(ifound == this_ibin)
            iselect = ix[-1]

            # Map to y value at specific x index in the bin:
            if select == "shallow":
                iselect = iselect[0]  # min/shallow
                mapped_value = y[iselect]
            elif select == "deep":
                iselect = iselect[-1]  # max/deep
                mapped_value = y[iselect]
            elif select == "middle":
                iselect = iselect[
                    np.where(x[iselect] >= np.median(x[iselect]))[0][0]
                ]  # median/middle
                mapped_value = y[iselect]
            elif select == "random":
                iselect = iselect[np.random.randint(len(iselect))]
                mapped_value = y[iselect]

            # or Map to y statistics in the bin:
            elif select == "mean":
                mapped_value = np.nanmean(y[iselect])
            elif select == "min":
                mapped_value = np.nanmin(y[iselect])
            elif select == "max":
                mapped_value = np.nanmax(y[iselect])
            elif select == "median":
                mapped_value = np.median(y[iselect])

            else:
                raise InvalidOption("`select` option has invalid value (%s)" % select)

            y_binned[this_ibin] = mapped_value

        return y_binned

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that don't contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a dataset input just apply the function for each appropriate array

    if version.parse(xr.__version__) > version.parse("0.15.0"):
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
            },
        )
    else:
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
        )
    remapped = xr.apply_ufunc(_subsample_bins, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped
