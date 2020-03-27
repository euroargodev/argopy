#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 12/03/2020

import os
import sys
import warnings
import requests
import io
from IPython.core.display import display, HTML
# import errno

def urlopen(url):
    """ Load content from url or raise alarm on status with explicit information on the error

    """
    # https://github.com/ioos/erddapy/blob/3828a4f479e7f7653fb5fd78cbce8f3b51bd0661/erddapy/utilities.py#L37
    r = requests.get(url)
    data = io.BytesIO(r.content)

    if r.status_code == 200:  # OK
        return data

    # 4XX client error response
    elif r.status_code == 404:  # Empty response
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8").replace("Error", ""))
        error.append("%s" % url)
        raise requests.HTTPError("\n".join(error))

    # 5XX server error response
    elif r.status_code == 500:  # 500 Internal Server Error
        if "text/html" in r.headers.get('content-type'):
            display(HTML(data.read().decode("utf-8")))
        error = ["Error %i " % r.status_code]
        error.append(data.decode("utf-8"))
        error.append("%s" % url)
        raise requests.HTTPError("\n".join(error))
    else:
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8"))
        error.append("%s" % url)
        print("\n".join(error))
        r.raise_for_status()

def list_available_data_backends():
    """ List all available data fetchers """
    AVAILABLE_BACKENDS = {}
    try:
        from .data_fetchers import erddap as Erddap_Fetchers
        AVAILABLE_BACKENDS['erddap'] = Erddap_Fetchers
    except:
        warnings.warn("An error occured while loading the ERDDAP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    try:
        from .data_fetchers import localftp as LocalFTP_Fetchers
        AVAILABLE_BACKENDS['localftp'] = LocalFTP_Fetchers
    except:
        warnings.warn("An error occured while loading the local FTP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    return AVAILABLE_BACKENDS

def list_standard_variables():
    """ Return the list of variables for standard users
    """
    return ['DATA_MODE', 'LATITUDE', 'LONGITUDE', 'POSITION_QC', 'DIRECTION', 'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'PRES',
     'TEMP', 'PSAL', 'PRES_QC', 'TEMP_QC', 'PSAL_QC', 'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED',
     'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC', 'PRES_ADJUSTED_ERROR', 'TEMP_ADJUSTED_ERROR',
     'PSAL_ADJUSTED_ERROR', 'JULD', 'JULD_QC', 'TIME', 'TIME_QC']

def list_multiprofile_file_variables():
    """ Return the list of variables in a netcdf multiprofile file.

        This is for files created by GDAC under <DAC>/<WMO>/<WMO>_prof.nc
    """
    return ['CONFIG_MISSION_NUMBER',
     'CYCLE_NUMBER',
     'DATA_CENTRE',
     'DATA_MODE',
     'DATA_STATE_INDICATOR',
     'DATA_TYPE',
     'DATE_CREATION',
     'DATE_UPDATE',
     'DC_REFERENCE',
     'DIRECTION',
     'FIRMWARE_VERSION',
     'FLOAT_SERIAL_NO',
     'FORMAT_VERSION',
     'HANDBOOK_VERSION',
     'HISTORY_ACTION',
     'HISTORY_DATE',
     'HISTORY_INSTITUTION',
     'HISTORY_PARAMETER',
     'HISTORY_PREVIOUS_VALUE',
     'HISTORY_QCTEST',
     'HISTORY_REFERENCE',
     'HISTORY_SOFTWARE',
     'HISTORY_SOFTWARE_RELEASE',
     'HISTORY_START_PRES',
     'HISTORY_STEP',
     'HISTORY_STOP_PRES',
     'JULD',
     'JULD_LOCATION',
     'JULD_QC',
     'LATITUDE',
     'LONGITUDE',
     'PARAMETER',
     'PI_NAME',
     'PLATFORM_NUMBER',
     'PLATFORM_TYPE',
     'POSITIONING_SYSTEM',
     'POSITION_QC',
     'PRES',
     'PRES_ADJUSTED',
     'PRES_ADJUSTED_ERROR',
     'PRES_ADJUSTED_QC',
     'PRES_QC',
     'PROFILE_PRES_QC',
     'PROFILE_PSAL_QC',
     'PROFILE_TEMP_QC',
     'PROJECT_NAME',
     'PSAL',
     'PSAL_ADJUSTED',
     'PSAL_ADJUSTED_ERROR',
     'PSAL_ADJUSTED_QC',
     'PSAL_QC',
     'REFERENCE_DATE_TIME',
     'SCIENTIFIC_CALIB_COEFFICIENT',
     'SCIENTIFIC_CALIB_COMMENT',
     'SCIENTIFIC_CALIB_DATE',
     'SCIENTIFIC_CALIB_EQUATION',
     'STATION_PARAMETERS',
     'TEMP',
     'TEMP_ADJUSTED',
     'TEMP_ADJUSTED_ERROR',
     'TEMP_ADJUSTED_QC',
     'TEMP_QC',
     'VERTICAL_SAMPLING_SCHEME',
     'WMO_INST_TYPE']

