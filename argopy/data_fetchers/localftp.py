#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for a local copy of GDAC ftp.
#
# This is not intended to be used directly, only by the facade at fetchers.py
#
# Since the GDAC ftp ir organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.
#
# Created by gmaze on 18/03/2020

access_points = ['wmo']
exit_formats = ['xarray']
dataset_ids = ['phy', 'bgc']

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

class LocalFTPArgoDataFetcher(ABC):
    """ Manage access to Argo data from a local copy of GDAC ftp

    """
    pass