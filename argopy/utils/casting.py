import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import importlib
import json
import logging
from copy import deepcopy


log = logging.getLogger("argopy.utils.casting")

path2assets = importlib.util.find_spec(
    "argopy.static.assets"
).submodule_search_locations[0]

with open(os.path.join(path2assets, "data_types.json"), "r") as f:
    DATA_TYPES = json.load(f)


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

    def cast_this(da, type, exception_to_raise=None):
        """Low-level casting of DataArray values"""
        try:
            da = da.astype(type)
            da.attrs["casted"] = 1
        except Exception as e:
            if exception_to_raise is not None:
                if isinstance(e, exception_to_raise):
                    raise
            else:
                msg = ["Oops! %s occurred" % sys.exc_info()[0]]
                msg.append(
                    "Fail to cast %s[%s] from '%s' to %s"
                    % (da.name, da.dims, da.dtype, type)
                )
                try:
                    msg.append("Unique values:", np.unique(da))
                except Exception:
                    msg.append("Can't read unique values !")
                    pass
                log.debug("\n".join(msg))
        return da

    def cast_this_da(da, v):
        """Cast any Argo DataArray"""
        # print("Casting %s ..." % da.name)
        da.attrs["casted"] = 0

        if v in DATA_TYPES["data"]["str"] and da.dtype == "O":  # Object
            try:
                da = cast_this(da, str, exception_to_raise=UnicodeDecodeError)
            except UnicodeDecodeError:
                da = da.str.decode(encoding="unicode_escape")
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

            elif "conventions" in da.attrs and da.attrs["conventions"] == "ISO8601":
                da.values = pd.to_datetime(da.values, utc=True)
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
        if (
            overwrite
            or ("casted" in ds[v].attrs and ds[v].attrs["casted"] == 0)
            or (
                not overwrite
                and "casted" in ds[v].attrs
                and ds[v].attrs["casted"] == 1
                and ds[v].dtype == "O"
            )
        ):
            try:
                attrs = deepcopy(ds[v].attrs)
                encoding = deepcopy(ds[v].encoding)
                ds[v] = cast_this_da(ds[v], v)
                casted_result = ds[v].attrs["casted"]
                ds[v].attrs = attrs
                ds[v].attrs.update({"casted": casted_result})
                ds[v].encoding = encoding
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
        elif isinstance(obj, tuple):
            obj = [o for o in obj]
        else:
            obj = [obj]
    return obj
