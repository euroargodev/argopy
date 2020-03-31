# -*coding: UTF-8 -*-
"""

    Useful for documentation, to play with and to test argopy

    Data files hosted on companion repository: http://www.github.com/euroargodev/argopy-data

"""

import os
# from os.path import dirname, join
import numpy as np
import xarray as xr
# import git
from zipfile import ZipFile

# import hashlib
from urllib.request import urlretrieve

_default_cache_dir = os.sep.join(("~", ".argopy_tutorial_data"))

class ftpsample():
    def __init__(self):
        self.localpath = os.path.expanduser(os.path.sep.join(["~",".argopy_tutorial_data"]))
        pass

    def download(self):
        # Download data repo
        # https://github.com/euroargodev/argopy-data/archive/master.zip