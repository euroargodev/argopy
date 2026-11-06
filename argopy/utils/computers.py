"""
Generic Functions to mathematically or statistically compute something out of Argo xarray objects

Rq: For thematic computations, use a dedicated module.

Parameters
----------
xarray objects or list of xarray objects

Returns
-------
xarray objects or list of xarray objects

"""

import numpy as np
from scipy import interpolate
import xarray as xr
from packaging import version
import logging
from dataclasses import dataclass

try:
    import gsw

    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False


from argopy.errors import InvalidOption

log = logging.getLogger("argopy.utils.compute")


@dataclass
class Msg:
    points_missing: str = "Not enough points to work with, skip profile interpolation"
    unstable_pres: str = "Encounter profile with unstable pressure, skip profile interpolation"
    nodata_left: str = "No data left to interpolate after pre-processing, skip profile interpolation"

#
#  From xarrayutils : https://github.com/jbusecke/xarrayutils/blob/master/xarrayutils/vertical_coordinates.py
#  Direct integration of those 2 functions to minimize dependencies and possibility of tuning them to our needs
#


def _std_level_boxes(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wong et al. (2026) 'box' edges [low, high) for each standard level"""
    mid = (levels[1:] + levels[:-1]) / 2
    lo = np.concatenate([[levels[0] - (levels[1] - levels[0]) / 2], mid])
    hi = np.concatenate([mid, [levels[-1] + (levels[-1] - levels[-2]) / 2]])
    return lo, hi


def _linear_interp_legacy(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """:func:`scipy.interpolate.interp1d` interpolation of a profile

    This consumes one single profile data as low-level structures, like numpy 1D arrays

    Expected to be used by `linear_interpolation_remap` and called by :func:`xr.apply_ufunc`

    """
    # remove all nans from input x and y
    idx = np.logical_or(np.isnan(x), np.isnan(y))
    x = x[~idx]
    y = y[~idx]

    # Need at least 5 points in the profile to interpolate, otherwise, return NaNs
    if len(y) < 5:
        interpolated = np.empty(len(xi))
        interpolated[:] = np.nan
    else:
        # replace nans in xi without of bound Values (just in case)
        xi = np.where(
            ~np.isnan(xi), xi, np.nanmax(x) + 1
        )
        # Interpolate with fill value parameter to extend min pressure toward 0
        interpolated = interpolate.interp1d(
            x, y, bounds_error=False, fill_value=(y[0], y[-1])
        )(xi)

    return interpolated


def _linear_interp(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """:func:`scipy.interpolate.interp1d` interpolation of a profile

    This consumes one single profile data as low-level structures, like numpy 1D arrays

    Expected to be used by `linear_interpolation_remap` and called by :func:`xr.apply_ufunc`
    """
    yi_empty = np.full(
        len(xi), np.nan, dtype=np.float32
    )  # Output when something fails

    # 'mask' holds array index to keep for interpolation

    # Un-select nans from input x and y:
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))

    # Un-select negative pressures:
    mask = np.logical_and(mask, x > 0)

    # At least 5 points:
    if np.nonzero(mask)[0].shape[0] < 5:
        log.debug(Msg().points_missing)
        return yi_empty

    # Skip a profile without monotonically increasing Pressure
    # todo: add test to remove profile with a density inversions > 0.03kg/m3
    dx = np.diff(x)
    if np.any(dx < 0):
        log.debug(
            Msg().unstable_pres
        )
        return yi_empty

    # Apply mask:
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        log.debug(
            Msg().nodata_left
        )
        return yi_empty

    # Following the spirit of Wong et al. (2026) EasyOneArgoLite specs for clean-up:

    # Run interp
    yi = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))(xi)

    # Step 2 - remove interpolated/extrapolated points from step 1 where there are no original data within the "box"
    # Box of level i is [lo[i], hi[i]): lower boundary included, upper boundary excluded
    if xi.size > 1:
        lo, hi = _std_level_boxes(xi)
        # in_box[i, j] is True if observation j is in the box of level i
        in_box = (x[np.newaxis, :] >= lo[:, np.newaxis]) & (x[np.newaxis, :] < hi[:, np.newaxis]) # used broadcasting
        no_data_in_box = ~in_box.any(axis=1)
        yi[no_data_in_box] = np.nan

    # Step 3 - remove the extrapolated points from step 1 outside the deepest observed pressure plus 1%
    too_deep = xi > (np.nanmax(x) * 1.01)
    yi[too_deep] = np.nan

    # TODO - check with Guillaume about this point (should we follow the spirit of Wong et al. (2026) here as well ?)
    # Clean up interpolated points outside of the pressure range of the input profile end points
    #yi[np.argwhere(xi < np.nanmin(x))] = np.nan
    #yi[np.argwhere(xi > np.nanmax(x))] = np.nan

    return yi


def _pchip_interp(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """:func:`gsw.pchip_interp` interpolation of a profile

    This consumes one single profile data as low-level structures, like numpy 1D arrays

    Expected to be used by `pchip_interpolation_remap` and called by :func:`xr.apply_ufunc`

    Follows Wong et al. (2026) EasyOneArgo - Lite specs for clean-up.
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
        log.debug(Msg().points_missing)
        return yi_empty

    # Skip a profile without monotonically increasing Pressure
    # todo: add test to remove profile with a density inversions > 0.03kg/m3
    dx = np.diff(x)
    if np.any(dx < 0):
        log.debug(
            Msg().unstable_pres
        )
        return yi_empty

    # Apply mask:
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        log.debug(
            Msg().nodata_left
        )
        return yi_empty

    # Follow Wong et al. (2026) EasyOneArgoLite specs for clean-up:

    # Run interp
    yi = gsw.pchip_interp(x, y, xi, axis=0) # Step 1

    # Step 2 - remove interpolated/extrapolated points from step 1 where there are no original data within the "box"
    # Box of level i is [lo[i], hi[i]): lower boundary included, upper boundary excluded
    if xi.size > 1:
        lo, hi = _std_level_boxes(xi)
        # in_box[i, j] is True if observation j is in the box of level i
        in_box = (x[np.newaxis, :] >= lo[:, np.newaxis]) & (x[np.newaxis, :] < hi[:, np.newaxis]) # used broadcasting
        no_data_in_box = ~in_box.any(axis=1)
        yi[no_data_in_box] = np.nan

    # Step 3 - remove the extrapolated points from step 1 outside the deepest observed pressure plus 1%
    too_deep = xi > (np.nanmax(x) * 1.01)
    yi[too_deep] = np.nan

    return yi


def _mrst_pchip_interp(x: np.ndarray, sa: np.ndarray, ct: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`gsw.sa_ct_interp` interpolation of a profile

    This consumes one single profile data of SA/CT as low-level structures, like numpy 1D arrays

    Expected to be used by `mrstpchip_interpolation_remap` and called by :func:`xr.apply_ufunc`

    Follows Wong et al. (2026) EasyOneArgo - Lite specs for clean-up.
    """
    yi_empty = np.full(
        len(xi), np.nan, dtype=np.float32
    )  # Output when something fails

    # 'mask' holds array index to keep for interpolation

    # Un-select nans from input x and y:
    mask = np.logical_and(np.logical_and(~np.isnan(x), ~np.isnan(sa)), ~np.isnan(ct))

    # Check for Barker and McDougall (2020) requirements:
    # Un-select negative pressures:
    mask = np.logical_and(mask, x > 0)

    # At least 5 points:
    if np.nonzero(mask)[0].shape[0] < 5:
        log.debug(Msg().points_missing)
        return yi_empty, yi_empty

    # Skip a profile without monotonically increasing Pressure
    # todo: add test to remove profile with a density inversions > 0.03kg/m3
    dx = np.diff(x)
    if np.any(dx < 0):
        log.debug(
            Msg().unstable_pres
        )
        return yi_empty, yi_empty

    # Apply mask:
    x = x[mask]
    sa = sa[mask]
    ct = ct[mask]

    if len(x) == 0:
        log.debug(
            Msg().nodata_left
        )
        return yi_empty, yi_empty

    # Follow Wong et al. (2026) EasyOneArgoLite specs for clean-up:

    # Run interp: SA, CT, p, p_i, axis=0)
    sai, cti= gsw.sa_ct_interp(sa, ct, x, xi, axis=0) # Step 1

    # Remove interpolated/extrapolated points from step 1 where there are no original data within the "box"
    # Box of level i is [lo[i], hi[i]): lower boundary included, upper boundary excluded
    if xi.size > 1:
        lo, hi = _std_level_boxes(xi)
        # in_box[i, j] is True if observation j is in the box of level i
        in_box = (x[np.newaxis, :] >= lo[:, np.newaxis]) & (x[np.newaxis, :] < hi[:, np.newaxis]) # used broadcasting
        no_data_in_box = ~in_box.any(axis=1)
        cti[no_data_in_box] = np.nan
        sai[no_data_in_box] = np.nan

    # Step 3 - Remove the extrapolated points from step 1 outside the deepest observed pressure plus 1%
    too_deep = xi > (np.nanmax(x) * 1.01)
    cti[too_deep] = np.nan
    sai[too_deep] = np.nan

    return sai, cti


def linear_interpolation_remap(
    z,
    data,
    z_regridded,
    z_dim=None,
    z_regridded_dim="Z_LEVELS",
    output_dim="PRES_INTERPOLATED"
) -> xr.Dataset:
    """Vertical interpolation of one dataset variable, using 1-D linear interpolation

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
    z_regridded_dim: str, optional, default='Z_LEVELS'
        Name of the output vertical dimension.
    output_dim: str, optional, default='PRES_INTERPOLATED'
        Name of the new vertical dimension
    """
    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, z must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

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

    remapped = xr.apply_ufunc(_linear_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]

    return remapped


def pchip_interpolation_remap(
    z,
    data,
    z_regridded,
    z_dim=None,
    z_regridded_dim="Z_LEVELS",
    output_dim="PRES_INTERPOLATED"
):
    """Vertical interpolation of one dataset variable, using Piecewise Cubic Hermite Interpolating Polynomial

    Parameters
    ----------
    z: :class:`xarray.DataArray`
        Input vertical pressure axis (eg: ds['PRES'])
    data: :class:`xarray.DataArray`
        Input variable to interpolate (eg: ds['TEMP'])
    z_regridded: :class:`xarray.DataArray`
        Output vertical pressure axis (eg: ds['Z_LEVELS'])
    z_dim: str, optional, default = None
        Name of the input vertical dimension (typically "N_LEVELS").
        If set to None, it is inferred from the input vertical pressure axis.
    z_regridded_dim: str, optional, default='Z_LEVELS'
        Name of the output vertical dimension.
    output_dim: str, optional, default='PRES_INTERPOLATED'
        Name of the new vertical dimension
    """
    if not with_gsw:
        raise ValueError("The pchip interpolation method requires the gsw library")

    # Infer dimension from input:
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, z must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # Set kwargs for xarray.apply_ufunc miscellaneaous versions:
    input_core_dims = [[dim], [dim], [z_regridded_dim]]

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

    remapped = xr.apply_ufunc(
                _pchip_interp, z, data, z_regridded, **kwargs
    )

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]

    return remapped


def mrst_pchip_interpolation_remap(
    z,
    Sdata,
    Tdata,
    z_regridded,
    z_dim=None,
    z_regridded_dim="Z_LEVELS",
    output_dim="PRES_INTERPOLATED"
):
    """Vertical interpolation of temperature/salinity, using the MRST-PCHIP method

    MRST-PCHIP: Multiply-Rotated Salinity-Temperature PCHIP Method

    Parameters
    ----------
    z: :class:`xarray.DataArray`
        Input vertical pressure axis (eg: ds['PRES'])
    Sdata: :class:`xarray.DataArray`
        Input salinity variable to interpolate (eg: ds['PSAL'])
    Tdata: :class:`xarray.DataArray`
        Input temperature variable to interpolate (eg: ds['TEMP'])
    z_regridded: :class:`xarray.DataArray`
        Output vertical pressure axis (eg: ds['Z_LEVELS'])
    z_dim: str, optional, default = None
        Name of the input vertical dimension (typically "N_LEVELS").
        If set to None, it is inferred from the input vertical pressure axis.
    z_regridded_dim: str, optional, default='Z_LEVELS'
        Name of the output vertical dimension.
    output_dim: str, optional, default='PRES_INTERPOLATED'
        Name of the new vertical dimension
    """
    if not with_gsw:
        raise ValueError("The mrst-pchip interpolation method requires the gsw library")

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, z must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # Set kwargs for xarray.apply_ufunc miscellaneaous versions:
    input_core_dims = [[dim], [dim], [dim], [z_regridded_dim]]

    kwargs = dict(
        input_core_dims=input_core_dims,
        output_core_dims=[[output_dim], [output_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[Sdata.dtype, Tdata.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
        },
    )

    remapped = xr.apply_ufunc(
            _mrst_pchip_interp, z, Sdata, Tdata, z_regridded, **kwargs
    )

    remapped[0].coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]

    remapped[1].coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]

    return remapped


def groupby_remap(
    z,
    data,
    z_regridded,  # noqa C901
    z_dim=None,
    z_regridded_dim="regridded",
    output_dim="remapped",
    select="deep",
    right=False,
) -> xr.Dataset:
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
