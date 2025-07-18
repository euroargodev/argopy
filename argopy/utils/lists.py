import sys
import warnings
import importlib
import os
import json
from ..options import OPTIONS
from typing import List, Union

path2assets = importlib.util.find_spec(
    "argopy.static.assets"
).submodule_search_locations[0]


def subsample_list(original_list, N):
    if len(original_list) <= N:
        return original_list
    else:
        step = len(original_list) / N
        indices = [int(i * step) for i in range(N)]
        subsampled_list = [original_list[i] for i in indices]
        return subsampled_list


def list_available_data_src() -> dict:
    """List all available data sources"""
    sources = {}
    try:
        from ..data_fetchers import erddap_data as Erddap_Fetchers

        # Ensure we're loading the erddap data fetcher with the current options:
        Erddap_Fetchers.api_server_check = Erddap_Fetchers.api_server_check.replace(
            Erddap_Fetchers.api_server, OPTIONS["erddap"]
        )
        Erddap_Fetchers.api_server = OPTIONS["erddap"]

        sources["erddap"] = Erddap_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ERDDAP data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from ..data_fetchers import argovis_data as ArgoVis_Fetchers

        sources["argovis"] = ArgoVis_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ArgoVis data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from ..data_fetchers import gdac_data as GDAC_Fetchers

        # Ensure we're loading the gdac data fetcher with the current options:
        GDAC_Fetchers.api_server_check = OPTIONS["gdac"]
        GDAC_Fetchers.api_server = OPTIONS["gdac"]

        sources["gdac"] = GDAC_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the GDAC data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    # return dict(sorted(sources.items()))
    return sources


def list_available_index_src() -> dict:
    """List all available index sources"""
    sources = {}
    try:
        from ..data_fetchers import erddap_index as Erddap_Fetchers

        # Ensure we're loading the erddap data fetcher with the current options:
        Erddap_Fetchers.api_server_check = Erddap_Fetchers.api_server_check.replace(
            Erddap_Fetchers.api_server, OPTIONS["erddap"]
        )
        Erddap_Fetchers.api_server = OPTIONS["erddap"]

        sources["erddap"] = Erddap_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ERDDAP index fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from ..data_fetchers import gdac_index as GDAC_Fetchers

        # Ensure we're loading the gdac data fetcher with the current options:
        GDAC_Fetchers.api_server_check = OPTIONS["gdac"]
        GDAC_Fetchers.api_server = OPTIONS["gdac"]

        sources["gdac"] = GDAC_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the GDAC index fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    return sources


def list_multiprofile_file_variables() -> List[str]:
    """List of all 🟡 core + 🔵 deep variables that can be found in a multi-profile netcdf file

    This is for files created by GDAC under <DAC>/<WMO>/<WMO>_prof.nc
    """
    return [
        "CONFIG_MISSION_NUMBER",
        "CYCLE_NUMBER",
        "DATA_CENTRE",
        "DATA_MODE",
        "DATA_STATE_INDICATOR",
        "DATA_TYPE",
        "DATE_CREATION",
        "DATE_UPDATE",
        "DC_REFERENCE",
        "DIRECTION",
        "FIRMWARE_VERSION",
        "FLOAT_SERIAL_NO",
        "FORMAT_VERSION",
        "HANDBOOK_VERSION",
        "HISTORY_ACTION",
        "HISTORY_DATE",
        "HISTORY_INSTITUTION",
        "HISTORY_PARAMETER",
        "HISTORY_PREVIOUS_VALUE",
        "HISTORY_QCTEST",
        "HISTORY_REFERENCE",
        "HISTORY_SOFTWARE",
        "HISTORY_SOFTWARE_RELEASE",
        "HISTORY_START_PRES",
        "HISTORY_STEP",
        "HISTORY_STOP_PRES",
        "JULD",
        "JULD_LOCATION",
        "JULD_QC",
        "LATITUDE",
        "LONGITUDE",
        "PARAMETER",
        "PI_NAME",
        "PLATFORM_NUMBER",
        "PLATFORM_TYPE",
        "POSITIONING_SYSTEM",
        "POSITION_QC",
        "PRES",
        "PRES_ADJUSTED",
        "PRES_ADJUSTED_ERROR",
        "PRES_ADJUSTED_QC",
        "PRES_QC",
        "PROFILE_PRES_QC",
        "PROFILE_PSAL_QC",
        "PROFILE_TEMP_QC",
        "PROJECT_NAME",
        "PSAL",
        "PSAL_ADJUSTED",
        "PSAL_ADJUSTED_ERROR",
        "PSAL_ADJUSTED_QC",
        "PSAL_QC",
        "REFERENCE_DATE_TIME",
        "SCIENTIFIC_CALIB_COEFFICIENT",
        "SCIENTIFIC_CALIB_COMMENT",
        "SCIENTIFIC_CALIB_DATE",
        "SCIENTIFIC_CALIB_EQUATION",
        "STATION_PARAMETERS",
        "TEMP",
        "TEMP_ADJUSTED",
        "TEMP_ADJUSTED_ERROR",
        "TEMP_ADJUSTED_QC",
        "TEMP_QC",
        "VERTICAL_SAMPLING_SCHEME",
        "WMO_INST_TYPE",
    ]


def list_core_parameters() -> List[str]:
    """List of all 🟡 core + 🔵 deep parameters that can be found in mono and multi-profile netcdf files

    This list is restricted to PARAMETERs for which the following variables can be found:

    - <PARAMETER>_DATA_MODE,
    - <PARAMETER>_QC,
    - <PARAMETER>_ADJUSTED,
    - <PARAMETER>_ADJUSTED_QC
    - <PARAMETER>_ADJUSTED_ERROR

    Returns
    -------
    List[str]

    """
    return ["PRES", "TEMP", "PSAL"]


def list_standard_variables(ds: str = 'phy') -> List[str]:
    """List of dataset variables possibly return in ``standard`` user mode

    Parameters
    ----------
    ds: str, default='phy'

        Return variables for one of the argopy ``ds`` option possible values:

         - ``phy``  is valid for the 🟡 core and 🔵 deep missions variables
         - ``bgc``  is valid for the 🟢 BGC missions variables

    Returns
    -------
    List[str]
    """

    # List of coordinates and meta-data to preserve in ``standard`` user mode:
    sv = [
        "LATITUDE",
        "LONGITUDE",
        "POSITION_QC",

        "DIRECTION",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        # "CONFIG_MISSION_NUMBER",

        "JULD",
        "JULD_QC",
        "TIME",
        "TIME_QC",
    ]

    if ds == 'phy':
        parameters = list_core_parameters()
        sv.append("DATA_MODE")
    elif ds in ['bgc', 'bgc-s']:
        parameters = list_bgc_s_parameters()

    for param in parameters:
        sv.append(param)
        if ds in ['bgc', 'bgc-s']:
            sv.append("%s_DATA_MODE" % param)
        sv.append("%s_QC" % param)
        sv.append("%s_ERROR" % param)   # <PARAM>_ERROR variables are added by :class:`Dataset.argo.datamode.merge`

        sv.append("%s_ADJUSTED" % param)
        sv.append("%s_ADJUSTED_QC" % param)
        sv.append("%s_ADJUSTED_ERROR" % param)

    return sv


def list_bgc_s_variables() -> List[str]:
    """List of all 🟢 BGC mission variables that can be found in a BGC **Synthetic** netcdf files

    This list includes (*but is not limited to*) PARAMETERs for which the following variables can be found:

    - <PARAMETER>_DATA_MODE,
    - <PARAMETER>_QC,
    - <PARAMETER>_ADJUSTED,
    - <PARAMETER>_ADJUSTED_QC
    - <PARAMETER>_ADJUSTED_ERROR

    This list also includes coordinates meta-data variables like LATITUDE or CONFIG_MISSION_NUMBER

    Returns
    -------
    List[str]

    See Also
    --------
    :meth:`argopy.utils.list_standard_variables`,
    :meth:`argopy.utils.list_bgc_s_parameters`,
    :meth:`argopy.utils.list_radiometry_variables`
    :meth:`argopy.utils.list_radiometry_parameters`,
    """
    with open(os.path.join(path2assets, "variables_bgc_synthetic.json"), "r") as f:
        vlist = json.load(f)
    return vlist["data"]["variables"]


def list_bgc_s_parameters() -> List[str]:
    """List of all 🟢 BGC mission parameters that can be found in a BGC **Synthetic** netcdf files

    This list is **restricted** to PARAMETERs for which the following variables can be found:

    - <PARAMETER>_DATA_MODE,
    - <PARAMETER>_QC,
    - <PARAMETER>_ADJUSTED,
    - <PARAMETER>_ADJUSTED_QC
    - <PARAMETER>_ADJUSTED_ERROR

    Returns
    -------
    List[str]

    See Also
    --------
    :meth:`argopy.utils.list_standard_variables`,
    :meth:`argopy.utils.list_bgc_s_variables`,
    :meth:`argopy.utils.list_radiometry_variables`
    :meth:`argopy.utils.list_radiometry_parameters`,
    """
    misc_meta = [
        "LATITUDE",
        "LONGITUDE",
        "DIRECTION",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "JULD",
        "TIME",
        "CONFIG_MISSION_NUMBER",
        "DATA_CENTRE",
        "DATA_TYPE",
        "DATE_UPDATE",
        "PI_NAME",
        "PLATFORM_TYPE",
        "WMO_INST_TYPE",
    ]
    return [
        v
        for v in list_bgc_s_variables()
        if "DATA_MODE" not in v
        and "QC" not in v
        and "ADJUSTED" not in v
        and v not in misc_meta
    ]


def list_radiometry_variables() -> List[str]:
    """List of all 🟢 BGC mission variables related to **radiometry** that can be found in a BGC **Synthetic** netcdf files

    This is a subset of the list returned by :meth:`argopy.utils.list_bgc_s_variables`.

    This list includes (but is not limited to) PARAMETERs for which the following variables can be found:

    - <PARAMETER>_DATA_MODE,
    - <PARAMETER>_QC,
    - <PARAMETER>_ADJUSTED,
    - <PARAMETER>_ADJUSTED_QC
    - <PARAMETER>_ADJUSTED_ERROR

    Returns
    -------
    List[str]

    See Also
    --------
    :meth:`argopy.utils.list_standard_variables`,
    :meth:`argopy.utils.list_bgc_s_variables`
    :meth:`argopy.utils.list_bgc_s_parameters`,
    :meth:`argopy.utils.list_radiometry_parameters`,
    """
    bgc_vlist_erddap = list_bgc_s_variables()
    vlist = []
    [vlist.append(v) for v in bgc_vlist_erddap if "up_radiance" in v.lower()]
    [vlist.append(v) for v in bgc_vlist_erddap if "down_irradiance" in v.lower()]
    [vlist.append(v) for v in bgc_vlist_erddap if "downwelling_par" in v.lower()]
    vlist.sort()
    return vlist


def list_radiometry_parameters() -> List[str]:
    """List of all 🟢 BGC mission parameters related to **radiometry** that can be found in a BGC **Synthetic** netcdf files

    This is a subset of the list returned by :meth:`argopy.utils.list_radiometry_variables`.

    This list is restricted to PARAMETERs for which the following variables can be found:

    - <PARAMETER>_DATA_MODE,
    - <PARAMETER>_QC,
    - <PARAMETER>_ADJUSTED,
    - <PARAMETER>_ADJUSTED_QC
    - <PARAMETER>_ADJUSTED_ERROR

    Returns
    -------
    List[str]

    See Also
    --------
    :meth:`argopy.utils.list_standard_variables`,
    :meth:`argopy.utils.list_bgc_s_variables`,
    :meth:`argopy.utils.list_bgc_s_parameters`
    :meth:`argopy.utils.list_radiometry_variables`,
    """
    params = list_radiometry_variables()
    return [
        v
        for v in params
        if "DATA_MODE" not in v and "QC" not in v and "ADJUSTED" not in v
    ]


def list_gdac_servers() -> List[str]:
    """List of official Argo GDAC servers

    Returns
    -------
    List[str]

    See also
    --------
    :class:`argopy.gdacfs`, :meth:`argopy.utils.check_gdac_path`, :meth:`argopy.utils.shortcut2gdac`

    """
    with open(os.path.join(path2assets, "gdac_servers.json"), "r") as f:
        vlist = json.load(f)
    return vlist["data"]["paths"]


def shortcut2gdac(short: str = None) -> Union[str, dict]:
    """Shortcut to GDAC server host mapping

    Parameters
    ----------
    short : str, optional
        Return GDAC host for a given shortcut, otherwise return the complete dictionary mapping. If the
        shortcut is unknown, return string unchanged.

    Returns
    -------
    str or dict

    See also
    --------
    :func:`argopy.utils.list_gdac_servers`, :class:`argopy.gdacfs`, :meth:`argopy.utils.check_gdac_path`

    """
    with open(os.path.join(path2assets, "gdac_servers.json"), "r") as f:
        vlist = json.load(f)
    shortcuts = vlist["data"]["shortcuts"]

    if short is not None:
        if short.lower().strip() in shortcuts.keys():
            return shortcuts[short.lower().strip()]
        else:
            return short
        # elif short in shortcuts.values():
        #     return short
        # else:
        #     raise ValueError("This shortcut '%s' does not exist. Must be one in [%s]" % (short, shortcuts.keys()))
    else:
        return shortcuts
