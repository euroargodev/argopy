import numpy as np
import pandas as pd
import xarray as xr
import getpass


def pre_process_multiprof(
    ds: xr.Dataset,
    access_point: str,
    access_point_opts: {},
    pre_filter_points: bool = False,
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

    # Remove raw netcdf file attributes and replace them with argopy ones:
    raw_attrs = ds.attrs
    ds.attrs = {}
    ds.attrs.update({"raw_attrs": raw_attrs})

    # Rename JULD and JULD_QC to TIME and TIME_QC
    ds = ds.rename(
        {"JULD": "TIME", "JULD_QC": "TIME_QC", "JULD_LOCATION": "TIME_LOCATION"}
    )
    ds["TIME"].attrs = {
        "long_name": "Datetime (UTC) of the station",
        "standard_name": "time",
    }

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

    ds = (
        ds.argo.profile2point()
    )  # Default output is a collection of points, along N_POINTS


    # Attributes are added by the caller

    # if dataset_id == "phy":
    #     ds.attrs["DATA_ID"] = "ARGO"
    # if dataset_id in ["bgc", "bgc-s"]:
    #     ds.attrs["DATA_ID"] = "ARGO-BGC"
    #
    # ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
    #
    # # ds.attrs["Fetched_from"] = server
    # ds.attrs["Fetched_constraints"] = cname
    # try:
    #     ds.attrs["Fetched_by"] = getpass.getuser()
    # except:  # noqa: E722
    #     ds.attrs["Fetched_by"] = "anonymous"
    # ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
    # ds.attrs["Fetched_uri"] = ds.encoding["source"]
    ds = ds[np.sort(ds.data_vars)]

    if pre_filter_points:
        ds = filter_points(ds, access_point=access_point, **access_point_opts)

    return ds


def filter_points(ds: xr.Dataset, access_point: str = None, **kwargs) -> xr.Dataset:
    """Enforce request criteria

    This may be necessary if for download performance improvement we had to work with multi instead of mono profile
    files: we loaded and merged multi-profile files, and then we need to make sure to retain only profiles requested.
    """
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
            np.zeros_like(ds["N_POINTS"]),
            dims=["N_POINTS"],
            coords={"N_POINTS": ds["N_POINTS"]},
        )
        for cyc in kwargs["CYC"]:
            this_mask += ds["CYCLE_NUMBER"] == cyc
        this_mask = this_mask >= 1  # any
        ds = ds.where(this_mask, drop=True)

    ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))

    return ds
