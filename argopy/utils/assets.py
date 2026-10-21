from typing import Any, Literal
import pandas as pd
from pathlib import Path
import importlib
import json
from functools import lru_cache

from argopy.errors import DataNotFound


class Asset:
    """Internal asset loader

    Assets are loaded using an instance of :class:`argopy.stores.filestore`.

    Notes
    -----
    This is **single-instance** class, whereby a single instance will be created during a session, whatever the number of calls is made. This avoids to create too many, and unnecessary, instances of file stores.

    Examples
    --------
    .. code-block:: python
        :caption: Examples of asset files loading

        Asset.load('data_types')
        Asset.load('data_types.json')
        Asset.load('schema:argo.float.schema')
        Asset.load('canyon-b:wgts_AT.txt', header=None, sep="\t")
    """

    _instance: "Asset | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "Asset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        if not self._initialized:
            path2assets = importlib.util.find_spec(
                "argopy.static.assets"
            ).submodule_search_locations[0]
            self._path = Path(path2assets)
            self._initialized = True

    @lru_cache
    def _read_csv(self, path, **kwargs):
        """Return a pandas.dataframe from a path that is a csv resource

        Parameters
        ----------
        Path: str
            Path to csv resources passed to :func:`pandas.read_csv`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        with open(path, 'r') as of:
            df = pd.read_csv(of, **kwargs)
        return df

    @lru_cache
    def _open_json(self, url, errors: Literal['raise', 'silent', 'ignore'] = 'raise', **kwargs) -> Any:
        """Open and process a json document from a path

        Steps performed:

        1. Path is open from ``url`` with :class:`filestore.open` and then
        2. Create a JSON with :func:`json.loads`.

        Each steps can be passed specifics arguments (see Parameters below).

        Parameters
        ----------
        path: str
            Path to resources passed to :func:`json.loads`
        errors: str, default: ``raise``
            Define how to handle errors:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console and return None
                - ``silent``:  Do not stop processing and do not issue log message, return None

        kwargs: dict

            - ``open_opts`` key dictionary is passed to :class:`open`
            - ``js_opts`` key dictionary is passed to :func:`json.loads`

        Returns
        -------
        Any

        See Also
        --------
        :class:`filestore.open_mfjson`
        """
        js_opts = {}
        if "js_opts" in kwargs:
            js_opts.update(kwargs["js_opts"])

        with open(url, 'r') as of:
            js = json.load(of, **js_opts)

        if len(js) == 0:
            if errors == "raise":
                raise DataNotFound("No data return by %s" % url)
            else:
                return None

        return js

    def _load(self, name: str, **kwargs) -> dict | pd.DataFrame:
        suffix = Path(name).suffix
        if suffix in [".csv", ".txt"]:
            load = self._read_csv
        else:
            load = self._open_json
            if suffix != ".json":  # eg: '.schema'
                name = f"{name}.json"

        name = name.strip()
        name = name.split(":")
        return load(self._path.joinpath(*name), **kwargs)

    @classmethod
    def load(cls, name: str = None, **kwargs) -> Any:
        """Load an asset file

        Parameters
        ----------
        name: str
            The *name* of the asset file to load.
            If no suffix is indicated, it is assumed to be a JSON file with a `.json` extension.
            If the asset is in sub-folders, use semicolons ':' as separator (eg: 'schema:argo.float.schema')
        **kwargs:
            All other arguments are passed down to the loading method.

        Notes
        -----
        If the asset `name` has a `.txt` or `.csv` suffix, the :meth:`argopy.stores.filestore.read_csv` is used.

        For all other asset `name`, the :meth:`argopy.stores.filestore.load_json` is used by default.
        """
        return cls()._load(name=name, **kwargs)
