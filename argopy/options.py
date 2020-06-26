# -*- coding: utf-8 -*-

"""
argopy.options
~~~~~~~~~~~~~~

This module manage options of the package

# Like always, largely inspired by xarray code:
# https://github.com/pydata/xarray/blob/cafab46aac8f7a073a32ec5aa47e213a9810ed54/xarray/core/options.py
"""
import os

# Define option names as seen by users:
DATA_SOURCE = 'src'
LOCAL_FTP = 'local_ftp'
DATASET = 'dataset'
DATA_CACHE = 'cachedir'
USER_LEVEL = 'mode'

# Define the list of available options and default values:
OPTIONS = {DATA_SOURCE: 'erddap',
           LOCAL_FTP: '.',
           DATASET: 'phy',
           DATA_CACHE: os.path.expanduser(os.path.sep.join(["~", ".cache", "argopy"])),
           USER_LEVEL: 'standard'}

# Define the list of possible values
_DATA_SOURCE_LIST = frozenset(["erddap", "localftp", "argovis"])
_DATASET_LIST = frozenset(["phy", "bgc", "ref"])
_USER_LEVEL_LIST = frozenset(["standard", "expert"])


# Define how to validate options:
def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    DATA_SOURCE: _DATA_SOURCE_LIST.__contains__,
    LOCAL_FTP: os.path.exists,
    DATASET: _DATASET_LIST.__contains__,
    USER_LEVEL: _USER_LEVEL_LIST.__contains__
}


# Implement the option setter:
class set_options:
    """Set options for argopy.

    List of options:

    - `dataset`: Dataset. This can be `phy`, `bgc` or `ref`.
      Default: `phy`

    - `src`: Source of fetched data.
      Default: `erddap`

    - `local_ftp`: Absolute path to local GDAC ftp copy.
      Default: `.`

    - `cachedir`: Absolute path to local cache directory.
      Default: `~/.cache/argopy`

    - `mode`: User mode. This can be `standard` or `expert`.
      Default: `standard`

    You can use `set_options` either as a context manager:
    ```
        >>> import argopy
        >>> with argopy.set_options(src='localftp'):
        ...     ds = argopy.DataFetcher().float(3901530).to_xarray()
    ```
    Or to set global options:
    ```
        >>> argopy.set_options(src='localftp')
    ```
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
