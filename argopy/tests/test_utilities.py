import os
import pytest
import tempfile
import xarray as xr
import pandas as pd
import numpy as np
import types

import argopy
from argopy.utilities import (
    linear_interpolation_remap,
    format_oneline,
    wmo2box,
    wrap_longitude,
    toYearFraction, YearFraction_to_datetime,
    argo_split_path,
    get_coriolis_profile_id,
    get_ea_profile_page,
)
from argopy.utils import (
    is_box,
    is_list_of_strings,
)
from argopy.errors import InvalidFetcherAccessPoint, FtpPathError
from argopy import DataFetcher as ArgoDataFetcher
from utils import (
    requires_connection,
    requires_erddap,
    requires_gdac,
)
from mocked_http import mocked_httpserver, mocked_server_address




