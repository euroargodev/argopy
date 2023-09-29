import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import importlib
import json
import logging

from .decorators import deprecated


log = logging.getLogger("argopy.utils.casting")

path2assets = importlib.util.find_spec(
    "argopy.static.assets"
).submodule_search_locations[0]

with open(os.path.join(path2assets, "data_types.json"), "r") as f:
    DATA_TYPES = json.load(f)


@deprecated("The 'cast_types' utility is deprecated since 0.1.13. It's been replaced by 'cast_Argo_variable_type'. Calling it will raise an error after argopy 0.1.16")
def cast_types(ds):  # noqa: C901
    """Make sure variables are of the appropriate types according to Argo

    #todo: This is hard coded, but should be retrieved from an API somewhere.
    Should be able to handle all possible variables encountered in the Argo dataset.

    Parameter
    ---------
    :class:`xarray.DataSet`

    Returns
    -------
    :class:`xarray.DataSet`
    """

    list_str = [
        "PLATFORM_NUMBER",
        "DATA_MODE",
        "DIRECTION",
        "DATA_CENTRE",
        "DATA_TYPE",
        "FORMAT_VERSION",
        "HANDBOOK_VERSION",
        "PROJECT_NAME",
        "PI_NAME",
        "STATION_PARAMETERS",
        "DATA_CENTER",
        "DC_REFERENCE",
        "DATA_STATE_INDICATOR",
        "PLATFORM_TYPE",
        "FIRMWARE_VERSION",
        "POSITIONING_SYSTEM",
        "PROFILE_PRES_QC",
        "PROFILE_PSAL_QC",
        "PROFILE_TEMP_QC",
        "PARAMETER",
        "SCIENTIFIC_CALIB_EQUATION",
        "SCIENTIFIC_CALIB_COEFFICIENT",
        "SCIENTIFIC_CALIB_COMMENT",
        "HISTORY_INSTITUTION",
        "HISTORY_STEP",
        "HISTORY_SOFTWARE",
        "HISTORY_SOFTWARE_RELEASE",
        "HISTORY_REFERENCE",
        "HISTORY_QCTEST",
        "HISTORY_ACTION",
        "HISTORY_PARAMETER",
        "VERTICAL_SAMPLING_SCHEME",
        "FLOAT_SERIAL_NO",
        "SOURCE",
        "EXPOCODE",
        "QCLEVEL",
    ]
    list_int = [
        "PLATFORM_NUMBER",
        "WMO_INST_TYPE",
        "WMO_INST_TYPE",
        "CYCLE_NUMBER",
        "CONFIG_MISSION_NUMBER",
    ]
    list_datetime = [
        "REFERENCE_DATE_TIME",
        "DATE_CREATION",
        "DATE_UPDATE",
        "JULD",
        "JULD_LOCATION",
        "SCIENTIFIC_CALIB_DATE",
        "HISTORY_DATE",
        "TIME",
    ]

    def fix_weird_bytes(x):
        x = x.replace(b"\xb1", b"+/-")
        return x

    fix_weird_bytes = np.vectorize(fix_weird_bytes)

    def cast_this(da, type):
        """Low-level casting of DataArray values"""
        try:
            da.values = da.values.astype(type)
            da.attrs["casted"] = 1
        except Exception:
            msg = (
                "Oops! %s occurred. Fail to cast <%s> into %s for: %s. Encountered unique values: %s"
                % (sys.exc_info()[0], str(da.dtype), type, da.name, str(np.unique(da)))
            )
            log.debug(msg)
        return da

    def cast_this_da(da):
        """Cast any DataArray"""
        v = da.name
        da.attrs["casted"] = 0
        if v in list_str and da.dtype == "O":  # Object
            if v in ["SCIENTIFIC_CALIB_COEFFICIENT"]:
                da.values = fix_weird_bytes(da.values)
            da = cast_this(da, str)

        if v in list_int:  # and da.dtype == 'O':  # Object
            da = cast_this(da, np.int32)

        if v in list_datetime and da.dtype == "O":  # Object
            if (
                "conventions" in da.attrs
                and da.attrs["conventions"] == "YYYYMMDDHHMISS"
            ):
                if da.size != 0:
                    if len(da.dims) <= 1:
                        val = da.astype(str).values.astype("U14")
                        # This should not happen, but still ! That's real world data
                        val[val == "              "] = "nan"
                        da.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                    else:
                        s = da.stack(dummy_index=da.dims)
                        val = s.astype(str).values.astype("U14")
                        # This should not happen, but still ! That's real world data
                        val[val == ""] = "nan"
                        val[val == "              "] = "nan"
                        #
                        s.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                        da.values = s.unstack("dummy_index")
                    da = cast_this(da, "datetime64[s]")
                else:
                    da = cast_this(da, "datetime64[s]")

            elif v == "SCIENTIFIC_CALIB_DATE":
                da = cast_this(da, str)
                s = da.stack(dummy_index=da.dims)
                s.values = pd.to_datetime(s.values, format="%Y%m%d%H%M%S")
                da.values = (s.unstack("dummy_index")).values
                da = cast_this(da, "datetime64[s]")

        if "QC" in v and "PROFILE" not in v and "QCTEST" not in v:
            if da.dtype == "O":  # convert object to string
                da = cast_this(da, str)

            # Address weird string values:
            # (replace missing or nan values by a '0' that will be cast as an integer later

            if da.dtype == "<U3":  # string, len 3 because of a 'nan' somewhere
                ii = (
                    da == "   "
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == "nan"
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                # Get back to regular U1 string
                da = cast_this(da, np.dtype("U1"))

            if da.dtype == "<U1":  # string
                ii = (
                    da == ""
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == " "
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == "n"
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

            # finally convert QC strings to integers:
            da = cast_this(da, np.int32)

        if da.dtype == "O":
            # By default, try to cast as float:
            da = cast_this(da, np.float32)

        if da.dtype != "O":
            da.attrs["casted"] = 1

        return da

    for v in ds.variables:
        try:
            ds[v] = cast_this_da(ds[v])
        except Exception:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Fail to cast: %s " % v)
            print("Encountered unique values:", np.unique(ds[v]))
            raise

    return ds


def cast_Argo_variable_type(ds, overwrite=True):
    """Ensure that all dataset variables are of the appropriate types according to Argo references

    Parameter
    ---------
    :class:`xarray.DataSet`
    overwrite: bool, default=True
        Should we force to re-cast a variable we already casted in this dataset ?

    Returns
    -------
    :class:`xarray.DataSet`
    """

    def cast_this(da, type):
        """Low-level casting of DataArray values"""
        try:
            da = da.astype(type)
            # with warnings.catch_warnings():
            #     warnings.filterwarnings('error')
            #     try:
            #         da = da.astype(type)
            #     except Warning:
            #         log.debug(type)
            #         log.debug(da.attrs)
            da.attrs["casted"] = 1
        except Exception:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print(
                "Fail to cast %s[%s] from '%s' to %s"
                % (da.name, da.dims, da.dtype, type)
            )
            try:
                print("Unique values:", np.unique(da))
            except Exception:
                print("Can't read unique values !")
                pass
        return da

    def cast_this_da(da, v):
        """Cast any Argo DataArray"""
        # print("Casting %s ..." % da.name)
        da.attrs["casted"] = 0

        if v in DATA_TYPES["data"]["str"] and da.dtype == "O":  # Object
            da = cast_this(da, str)

        if v in DATA_TYPES["data"]["int"]:  # and da.dtype == 'O':  # Object
            if "conventions" in da.attrs:
                convname = "conventions"
            elif "convention" in da.attrs:
                convname = "convention"
            else:
                convname = None
            if convname in da.attrs and da.attrs[convname] in [
                "Argo reference table 19",
                "Argo reference table 21",
                "WMO float identifier : A9IIIII",
                "1...N, 1 : first complete mission",
            ]:
                # Some values may be missing, and the _FillValue=" " cannot be casted as an integer.
                # so, we replace missing values with a 999:
                val = da.astype(str).values
                # val[np.where(val == 'nan')] = '999'
                val[val == "nan"] = "999"
                da.values = val
            da = cast_this(da, float)
            da = cast_this(da, int)

        if v in DATA_TYPES["data"]["datetime"] and da.dtype == "O":  # Object
            if (
                "conventions" in da.attrs
                and da.attrs["conventions"] == "YYYYMMDDHHMISS"
            ):
                if da.size != 0:
                    if len(da.dims) <= 1:
                        val = da.astype(str).values.astype("U14")
                        # This should not happen, but still ! That's real world data
                        val[val == "              "] = "nan"
                        da.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                    else:
                        s = da.stack(dummy_index=da.dims)
                        val = s.astype(str).values.astype("U14")
                        # This should not happen, but still ! That's real world data
                        val[val == "              "] = "nan"
                        s.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                        da.values = s.unstack("dummy_index")
                    da = cast_this(da, "datetime64[ns]")
                else:
                    da = cast_this(da, "datetime64[ns]")

            elif v == "SCIENTIFIC_CALIB_DATE":
                da = cast_this(da, str)
                s = da.stack(dummy_index=da.dims)
                s.values = pd.to_datetime(s.values, format="%Y%m%d%H%M%S")
                da.values = (s.unstack("dummy_index")).values
                da = cast_this(da, "datetime64[ns]")

        if "QC" in v and "PROFILE" not in v and "QCTEST" not in v:
            if da.dtype == "O":  # convert object to string
                da = cast_this(da, str)

            # Address weird string values:
            # (replace missing or nan values by a '0' that will be cast as an integer later
            if da.dtype == float:
                val = da.astype(str).values
                val[np.where(val == "nan")] = "0"
                da.values = val
                da = cast_this(da, float)

            if da.dtype == "<U3":  # string, len 3 because of a 'nan' somewhere
                ii = (
                    da == "   "
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == "nan"
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                # Get back to regular U1 string
                da = cast_this(da, np.dtype("U1"))

            if da.dtype == "<U1":  # string
                ii = (
                    da == ""
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == " "
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

                ii = (
                    da == "n"
                )  # This should not happen, but still ! That's real world data
                da = xr.where(ii, "0", da)

            # finally convert QC strings to integers:
            da = cast_this(da, int)

        if "DATA_MODE" in v:
            da = cast_this(da, "<U1")

        if da.dtype != "O":
            da.attrs["casted"] = 1

        return da

    for v in ds.variables:
        if overwrite or ("casted" in ds[v].attrs and ds[v].attrs["casted"] == 0):
            try:
                ds[v] = cast_this_da(ds[v], v)
            except Exception:
                print("Oops!", sys.exc_info()[0], "occurred.")
                print("Fail to cast: %s " % v)
                print("Encountered unique values:", np.unique(ds[v]))
                raise

    return ds


def to_list(obj):
    """Make sure that an expected list is indeed a list"""
    if not isinstance(obj, list):
        if isinstance(obj, np.ndarray):
            obj = list(obj)
        else:
            obj = [obj]
    return obj
