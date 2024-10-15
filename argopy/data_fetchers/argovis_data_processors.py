import pandas as pd
import xarray as xr
from typing import Any
from ..utils import list_bgc_s_parameters


def pre_process(profiles: Any, key_map: dict = None) -> pd.DataFrame:
    """convert json data to Pandas DataFrame"""

    if profiles is None:
        return None

    # Make sure we deal with a list
    if isinstance(profiles, list):
        data = profiles
    else:
        data = [profiles]

    # Transform
    rows = []
    for profile in data:
        # construct metadata dictionary that will be repeated for each level
        metadict = {
            "date": profile["timestamp"],
            "date_qc": profile["timestamp_argoqc"],
            "lat": profile["geolocation"]["coordinates"][1],
            "lon": profile["geolocation"]["coordinates"][0],
            "cycle_number": profile["cycle_number"],
            "DATA_MODE": profile["data_info"][2][0][1],
            "DIRECTION": profile["profile_direction"],
            "platform_number": profile["_id"].split("_")[0],
            "position_qc": profile["geolocation_argoqc"],
            "index": 0,
        }
        # construct a row for each level in the profile
        for i in range(
                len(profile["data"][profile["data_info"][0].index("pressure")])
        ):
            row = {
                "temp": profile["data"][
                    profile["data_info"][0].index("temperature")
                ][i],
                "pres": profile["data"][profile["data_info"][0].index("pressure")][
                    i
                ],
                "psal": profile["data"][profile["data_info"][0].index("salinity")][
                    i
                ],
                **metadict,
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    df = df.reset_index()
    if key_map is not None:
        df = df.rename(columns=key_map)
        df = df[[value for value in key_map.values() if value in df.columns]]

    return df


def add_attributes(this: xr.Dataset) -> xr.Dataset:  # noqa: C901
    """Add variables attributes not return by argovis requests

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

    if "TIME" in this.data_vars or "TIME" in this.coords:
        this["TIME"].attrs = {
            "long_name": "Datetime (UTC) of the station",
            "standard_name": "time",
            "axis": "T",
            "conventions": "ISO8601",
            "casted": this["TIME"].attrs["casted"]
            if "casted" in this["TIME"].attrs
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
