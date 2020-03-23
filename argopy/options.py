#!/bin/env python
# -*coding: UTF-8 -*-
#
# Manage options of the package
#

# Like always, largely inspired by xarray code.
# https://github.com/pydata/xarray/blob/cafab46aac8f7a073a32ec5aa47e213a9810ed54/xarray/core/options.py

# Define option names as seen by users:
DATA_FETCHER = 'data_src'
LOCAL_FTP = 'local_ftp'
DATASET = 'dataset'
DATA_CACHE = 'cachedir'

# Define the list of available options:
OPTIONS = {DATA_FETCHER: 'erddap',
           LOCAL_FTP: '.',
           DATASET: 'phy',
           DATA_CACHE: None
}

# Define the list of possible values
_DATA_FETCHER_LIST = frozenset(["erddap", "localftp"])
_DATASET_LIST = frozenset(["phy", "bgc", "ref"])

# Define how to validate options:
def _positive_integer(value):
    return isinstance(value, int) and value > 0

import os

_VALIDATORS = {
    DATA_FETCHER: _DATA_FETCHER_LIST.__contains__,
    LOCAL_FTP: os.path.exists,
    DATASET: _DATASET_LIST.__contains__
}

# Implement the option setter:
class set_options:
    """Set options for argopy.

    Lis of options:
    - ``dataset``: Dataset. This can be ``phy``, ``bgc`` or ``ref``.
      Default: ``phy``
    - ``data_fetcher``: Backend for fetching data.
      Default: ``erddap``.
    - ``local_ftp``: Absolute path to local GDAC ftp copy.
      Default: ``???``.

    You can use ``set`` either as a context manager:
    >>> import argopy
    >>> with argopy.set(data_fetcher='localftp'):
    ...     ds = argopy.DataFetcher().float(3901530).to_xarray()

    Or to set global options:
    >>> argopy.set(data_fetcher='localftp')
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError("argument name %r is not in the set of valid options %r" % (k, set(OPTIONS)))

            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        # for k, v in options_dict.items():
        #     if k in _SETTERS:
        #         _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)