import numpy as np
import pandas as pd
import xarray as xr
import getpass
from urllib.parse import urlparse
from ..utils import to_list, UriCName
from ..utils import list_bgc_s_parameters


def pre_process(
        this_ds, add_dm: bool = True, dataset_id: str = 'phy', **kwargs
):  # noqa: C901
    """Pre-processor of a xarray.DataSet created from a netcdf erddap response

    This method can also be applied on a regular dataset to re-enforce format compliance

    This methdd must be seriablizable:

        >>> from distributed.protocol import serialize
        >>> from distributed.protocol.serialize import ToPickle
        >>> serialize(ToPickle(post_process))


    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        Dataset to process

    Returns
    -------
    :class:`xarray.Dataset`
    """
    if this_ds is None:
        return None

    if "row" in this_ds.dims:
        this_ds = this_ds.rename({"row": "N_POINTS"})
    elif "N_POINTS" in this_ds.dims:
        this_ds["N_POINTS"] = np.arange(0, len(this_ds["N_POINTS"]))

    # Set coordinates:
    coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
    this_ds = this_ds.reset_coords()
    this_ds["N_POINTS"] = this_ds["N_POINTS"]

    # Convert all coordinate variable names to upper case
    for v in this_ds.data_vars:
        this_ds = this_ds.rename({v: v.upper()})
    this_ds = this_ds.set_coords(coords)

    if dataset_id == "ref":
        this_ds["DIRECTION"] = xr.full_like(this_ds["CYCLE_NUMBER"], "A", dtype=str)

    # Cast data types:
    this_ds = this_ds.argo.cast_types()

    # With BGC, some points may not have a PLATFORM_NUMBER !
    # So, we remove these
    if dataset_id in ["bgc", "bgc-s"] and "999" in to_list(
            np.unique(this_ds["PLATFORM_NUMBER"].values)
    ):
        this_ds = this_ds.where(this_ds["PLATFORM_NUMBER"] != "999", drop=True)
        this_ds = this_ds.argo.cast_types(overwrite=True)

    if dataset_id in ["bgc", "bgc-s"] and add_dm:
        this_ds = this_ds.argo.datamode.compute(indexfs=kwargs['indexfs'])
        this_ds = this_ds.argo.cast_types(overwrite=False)

    # Overwrite Erddap attributes with those from Argo standards:
    this_ds = _add_attributes(this_ds)

    # In the case of a parallel download, this is a trick to preserve the chunk uri in the chunk dataset:
    # (otherwise all chunks have the same list of uri)
    # "history" is an attribute return by the erddap
    if 'Fetched_uri' in this_ds.attrs:
        Fetched_url = this_ds.attrs.get("Fetched_uri")
    else:
        Fetched_url = this_ds.attrs.get("history", "").split('\n')[-1].split(' ')[-1]

    # Finally overwrite erddap attributes with those from argopy:
    raw_attrs = this_ds.attrs.copy()
    this_ds.attrs = {}
    if dataset_id == "phy":
        this_ds.attrs["DATA_ID"] = "ARGO"
        this_ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
    elif dataset_id == "ref":
        this_ds.attrs["DATA_ID"] = "Reference_ARGO_CTD"
        this_ds.attrs["DOI"] = "-"
        this_ds.attrs["Fetched_version"] = raw_attrs.get('version', '?')
    elif dataset_id == "ref-ctd":
        this_ds.attrs["DATA_ID"] = "Reference_SHIP_CTD"
        this_ds.attrs["DOI"] = "-"
        this_ds.attrs["Fetched_version"] = raw_attrs.get('version', '?')
    elif dataset_id in ["bgc", "bgc-s"]:
        this_ds.attrs["DATA_ID"] = "ARGO-BGC"
        this_ds.attrs["DOI"] = "http://doi.org/10.17882/42182"

    this_ds.attrs["Fetched_from"] = urlparse(Fetched_url).netloc
    try:
        this_ds.attrs["Fetched_by"] = getpass.getuser()
    except:  # noqa: E722
        this_ds.attrs["Fetched_by"] = "anonymous"
    this_ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime(
        "%Y/%m/%d"
    )
    this_ds.attrs["Fetched_constraints"] = UriCName(Fetched_url).cname
    this_ds.attrs["Fetched_uri"] = Fetched_url
    this_ds = this_ds[np.sort(this_ds.data_vars)]

    # if dataset_id in ["bgc", "bgc-s"]:
    #     n_zero = np.count_nonzero(np.isnan(np.unique(this_ds["PLATFORM_NUMBER"])))
    #     if n_zero > 0:
    #         log.error("Some points (%i) have no PLATFORM_NUMBER !" % n_zero)

    return this_ds


def _add_attributes(this):  # noqa: C901
    """Add variables attributes not return by erddap requests

    This is hard coded, but should be retrieved from an API somewhere
    """

    for v in this.data_vars:
        param = "PRES"
        if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
            this[v].attrs = {
                "long_name": "Sea Pressure",
                "standard_name": "sea_water_pressure",
                "units": "decibar",
                "valid_min": 0.0,
                "valid_max": 12000.0,
                "resolution": 0.1,
                "axis": "Z",
                "casted": this[v].attrs["casted"]
                if "casted" in this[v].attrs
                else 0,
            }
            if "ERROR" in v:
                this[v].attrs["long_name"] = (
                    "ERROR IN %s" % this[v].attrs["long_name"]
                )

    for v in this.data_vars:
        param = "TEMP"
        if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
            this[v].attrs = {
                "long_name": "SEA TEMPERATURE IN SITU ITS-90 SCALE",
                "standard_name": "sea_water_temperature",
                "units": "degree_Celsius",
                "valid_min": -2.0,
                "valid_max": 40.0,
                "resolution": 0.001,
                "casted": this[v].attrs["casted"]
                if "casted" in this[v].attrs
                else 0,
            }
            if "ERROR" in v:
                this[v].attrs["long_name"] = (
                    "ERROR IN %s" % this[v].attrs["long_name"]
                )

    for v in this.data_vars:
        param = "PSAL"
        if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
            this[v].attrs = {
                "long_name": "PRACTICAL SALINITY",
                "standard_name": "sea_water_salinity",
                "units": "psu",
                "valid_min": 0.0,
                "valid_max": 43.0,
                "resolution": 0.001,
                "casted": this[v].attrs["casted"]
                if "casted" in this[v].attrs
                else 0,
            }
            if "ERROR" in v:
                this[v].attrs["long_name"] = (
                    "ERROR IN %s" % this[v].attrs["long_name"]
                )

    for v in this.data_vars:
        param = "DOXY"
        if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
            this[v].attrs = {
                "long_name": "Dissolved oxygen",
                "standard_name": "moles_of_oxygen_per_unit_mass_in_sea_water",
                "units": "micromole/kg",
                "valid_min": -5.0,
                "valid_max": 600.0,
                "resolution": 0.001,
                "casted": this[v].attrs["casted"]
                if "casted" in this[v].attrs
                else 0,
            }
            if "ERROR" in v:
                this[v].attrs["long_name"] = (
                    "ERROR IN %s" % this[v].attrs["long_name"]
                )

    for v in this.data_vars:
        if "_QC" in v:
            attrs = {
                "long_name": "Global quality flag of %s profile" % v,
                "conventions": "Argo reference table 2a",
                "casted": this[v].attrs["casted"]
                if "casted" in this[v].attrs
                else 0,
            }
            this[v].attrs = attrs

    if "CYCLE_NUMBER" in this.data_vars:
        this["CYCLE_NUMBER"].attrs = {
            "long_name": "Float cycle number",
            "conventions": "0..N, 0 : launch cycle (if exists), 1 : first complete cycle",
            "casted": this["CYCLE_NUMBER"].attrs["casted"]
            if "casted" in this["CYCLE_NUMBER"].attrs
            else 0,
        }
    if "DIRECTION" in this.data_vars:
        this["DIRECTION"].attrs = {
            "long_name": "Direction of the station profiles",
            "conventions": "A: ascending profiles, D: descending profiles",
            "casted": this["DIRECTION"].attrs["casted"]
            if "casted" in this["DIRECTION"].attrs
            else 0,
        }

    if "PLATFORM_NUMBER" in this.data_vars:
        this["PLATFORM_NUMBER"].attrs = {
            "long_name": "Float unique identifier",
            "conventions": "WMO float identifier : A9IIIII",
            "casted": this["PLATFORM_NUMBER"].attrs["casted"]
            if "casted" in this["PLATFORM_NUMBER"].attrs
            else 0,
        }

    if "DATA_MODE" in this.data_vars:
        this["DATA_MODE"].attrs = {
            "long_name": "Delayed mode or real time data",
            "conventions": "R : real time; D : delayed mode; A : real time with adjustment",
            "casted": this["DATA_MODE"].attrs["casted"]
            if "casted" in this["DATA_MODE"].attrs
            else 0,
        }

    for param in list_bgc_s_parameters():
        if "%s_DATA_MODE" % param in this.data_vars:
            this["%s_DATA_MODE" % param].attrs = {
                "long_name": "Delayed mode or real time data",
                "conventions": "R : real time; D : delayed mode; A : real time with adjustment",
                "casted": this["%s_DATA_MODE" % param].attrs["casted"]
                if "casted" in this["%s_DATA_MODE" % param].attrs
                else 0,
            }

    if "LATITUDE" in this.data_vars or "LATITUDE" in this.coords:
        this["LATITUDE"].attrs = {
            "long_name": "Latitude of the station, best estimate",
            "standard_name": "latitude",
            "axis": "Y",
            "valid_min": -90.0,
            "valid_max": 90.0,
            "casted": this["LATITUDE"].attrs["casted"]
            if "casted" in this["LATITUDE"].attrs
            else 0,
        }

    if "LONGITUDE" in this.data_vars or "LONGITUDE" in this.coords:
        this["LONGITUDE"].attrs = {
            "long_name": "Longitude of the station, best estimate",
            "standard_name": "longitude",
            "axis": "X",
            "valid_min": -180.0,
            "valid_max": 180.0,
            "casted": this["LONGITUDE"].attrs["casted"]
            if "casted" in this["LONGITUDE"].attrs
            else 0,
        }

    return this
