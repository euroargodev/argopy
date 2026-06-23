import numpy as np
import xarray as xr
from typing import Literal
import logging

log = logging.getLogger("argopy.gdac.data")


def pre_process_multiprof(
    ds: xr.Dataset,
    access_point: str,
    access_point_opts: {},
    params_list: list = None,
    measured_params: list = None,
    pre_filter_points: bool = False,
    dimension: Literal["point", "profile"] = "point",
    # dataset_id: str = "phy",
    # cname: str = '?',
) -> xr.Dataset:
    """Pre-process one Argo multi-profile file as a collection of points

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        Dataset to process

    Returns
    -------
    :class:`xarray.Dataset`
    """
    if ds is None:
        return None

    # # Remove raw netcdf file attributes and replace them with argopy ones:
    # raw_attrs = ds.attrs
    # ds.attrs = {}
    # ds.attrs.update({"raw_attrs": raw_attrs})

    # Rename JULD and JULD_QC to TIME and TIME_QC
    ds = ds.rename(
        {"JULD": "TIME", "JULD_QC": "TIME_QC", "JULD_LOCATION": "TIME_LOCATION"}
    )
    ds["TIME"].attrs = {
        "long_name": "Datetime (UTC) of the station",
        "standard_name": "time",
    }

    # Ensure N_PROF is a coordinate
    # ds = ds.assign_coords(N_PROF=np.arange(0, len(ds["N_PROF"])))
    ds = ds.reset_coords()
    coords = ("LATITUDE", "LONGITUDE", "TIME", "N_PROF")
    ds = ds.assign_coords({"N_PROF": np.arange(0, len(ds["N_PROF"]))})
    ds = ds.set_coords(coords)

    # Cast data types:
    ds = ds.argo.cast_types()

    # Enforce real pressure resolution : 0.1 db
    for vname in ds.data_vars:
        if "PRES" in vname and "QC" not in vname:
            ds[vname].values = np.round(ds[vname].values, 1)

    # Remove variables without dimensions:
    # todo: We should be able to find a way to keep them somewhere in the data structure
    for v in ds.data_vars:
        if len(list(ds[v].dims)) == 0:
            ds = ds.drop_vars(v)

    if dimension == "point":
        ds = (
            ds.argo.profile2point()
        )  # Default output is a collection of points, along N_POINTS

    ds = ds[np.sort(ds.data_vars)]

    if pre_filter_points:
        ds = filter_points(ds, access_point=access_point, **access_point_opts)

    # keep only requested params if specified (for bgc datasets)
    if params_list is not None:
        params_list = [p for p in params_list if p in ds.data_vars]
        ds = ds[params_list]

    # Apply the 'measured' criteria for BGC requests:
    if measured_params is not None:
        ds = filter_measured(ds, measured_params=measured_params)

    return ds


def filter_measured(ds: xr.Dataset, measured_params: list = None) -> xr.Dataset:
    """Re-enforce the 'measured' criteria for BGC requests

    Parameters
    ----------
    ds: :class:`xr.Dataset`

    """
    # Enforce the 'measured' argument for BGC:
    if len(measured_params) == 0:
        return ds
    elif len(ds["N_POINTS"]) > 0:
        log.debug("Keep only samples without NaN in %s" % measured_params)
        for v in measured_params:
            this_mask = None
            if v in ds and "%s_ADJUSTED" % v in ds:
                this_mask = np.logical_or.reduce(
                    (ds[v].notnull(), ds["%s_ADJUSTED" % v].notnull())
                )
            elif v in ds:
                this_mask = ds[v].notnull()
            elif "%s_ADJUSTED" % v in ds:
                this_mask = ds["%s_ADJUSTED" % v].notnull()
            else:
                log.debug(
                    "'%s' or '%s_ADJUSTED' not in the dataset to apply the 'filter_measured' method"
                    % (v, v)
                )
            if this_mask is not None:
                ds = ds.loc[dict(N_POINTS=this_mask)]

    ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
    return ds


def filter_points(ds: xr.Dataset, access_point: str = None, **kwargs) -> xr.Dataset:
    """Enforce request criteria

    This may be necessary if for download performance improvement we had to work with multi instead of mono profile
    files: we loaded and merged multi-profile files, and then we need to make sure to retain only profiles requested.
    """
    dim = "N_PROF" if "N_PROF" in ds.dims else "N_POINTS"
    ds = ds.assign_coords({dim: np.arange(0, len(ds[dim]))})
    if "N_LEVELS" in ds.dims:
        ds = ds.assign_coords({"N_LEVELS": np.arange(0, len(ds["N_LEVELS"]))})

    if access_point == "BOX":
        BOX = kwargs["BOX"]
        # - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
        # - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        ds = (
            ds.where(ds["LONGITUDE"] >= BOX[0], drop=True)
            .where(ds["LONGITUDE"] < BOX[1], drop=True)
            .where(ds["LATITUDE"] >= BOX[2], drop=True)
            .where(ds["LATITUDE"] < BOX[3], drop=True)
            .where(ds["PRES"] >= BOX[4], drop=True)  # todo what about PRES_ADJUSTED ?
            .where(ds["PRES"] < BOX[5], drop=True)
        )
        if len(BOX) == 8:
            ds = ds.where(ds["TIME"] >= np.datetime64(BOX[6]), drop=True).where(
                ds["TIME"] < np.datetime64(BOX[7]), drop=True
            )

    if access_point == "CYC":
        this_mask = xr.DataArray(
            np.zeros_like(ds[dim]),
            dims=[dim],
            coords={dim: ds[dim]},
        )
        for cyc in kwargs["CYC"]:
            this_mask += ds["CYCLE_NUMBER"] == cyc
        this_mask = this_mask >= 1  # any
        ds = ds.where(this_mask, drop=True)

    ds = ds.assign_coords({dim: np.arange(0, len(ds[dim]))})
    return ds
