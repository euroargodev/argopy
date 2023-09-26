import sys
import warnings
from ..options import OPTIONS


def list_available_data_src():
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
        from ..data_fetchers import gdacftp_data as GDAC_Fetchers

        # Ensure we're loading the gdac data fetcher with the current options:
        GDAC_Fetchers.api_server_check = OPTIONS["ftp"]
        GDAC_Fetchers.api_server = OPTIONS["ftp"]

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


def list_available_index_src():
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
        from ..data_fetchers import gdacftp_index as GDAC_Fetchers

        # Ensure we're loading the gdac data fetcher with the current options:
        GDAC_Fetchers.api_server_check = OPTIONS["ftp"]
        GDAC_Fetchers.api_server = OPTIONS["ftp"]

        sources["gdac"] = GDAC_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the GDAC index fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    return sources


def list_standard_variables():
    """List of variables for standard users"""
    return [
        "DATA_MODE",
        "LATITUDE",
        "LONGITUDE",
        "POSITION_QC",
        "DIRECTION",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "PRES",
        "TEMP",
        "PSAL",
        "PRES_QC",
        "TEMP_QC",
        "PSAL_QC",
        "PRES_ADJUSTED",
        "TEMP_ADJUSTED",
        "PSAL_ADJUSTED",
        "PRES_ADJUSTED_QC",
        "TEMP_ADJUSTED_QC",
        "PSAL_ADJUSTED_QC",
        "PRES_ADJUSTED_ERROR",
        "TEMP_ADJUSTED_ERROR",
        "PSAL_ADJUSTED_ERROR",
        "PRES_ERROR",  # can be created from PRES_ADJUSTED_ERROR after a filter_data_mode
        "TEMP_ERROR",
        "PSAL_ERROR",
        "JULD",
        "JULD_QC",
        "TIME",
        "TIME_QC",
        # "CONFIG_MISSION_NUMBER",
    ]


def list_multiprofile_file_variables():
    """List of variables in a netcdf multiprofile file.

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
