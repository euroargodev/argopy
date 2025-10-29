from typing import Any, Callable
import inspect
from functools import lru_cache

from ...stores import httpstore
from ...options import OPTIONS


class NVS:
    """A class that will be instantiated only once to avoid multiple NVS store connectors

    Used by other classes to handle NVS json download
    """

    nvs: str = None
    """Url to the NVS"""

    uid: str = None
    """Unique instance ID"""

    _fs: Any = None
    _instance: "NVS | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "NVS":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        if not self._initialized:
            self._fs = httpstore(
                cache=kwargs.get("cache", True),
                cachedir=kwargs.get("cachedir", OPTIONS["cachedir"]),
                timeout=kwargs.get("timeout", OPTIONS["api_timeout"]),
            )
            self._initialized = True
        self.uid = id(self)
        self.nvs = kwargs.get("nvs", OPTIONS["nvs"])

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only after instantiation policy for public attributes"""
        if (
            attr in [key for key in self.__dir__() if key[0] != "_"]
            and inspect.stack()[1][3] != "__init__"
        ):
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        self.__dict__[f"{attr}"] = value

    def __repr__(self):
        props = [
            key
            for key in self.__dir__()
            if key[0] != "_" and not isinstance(getattr(self, key), Callable)
        ]
        props = sorted(props)
        props_str = [f"{prop}={getattr(self, prop)}" for prop in props]
        return f"NVS({', '.join(props_str)})"

    @lru_cache
    def open_json(self, *args, **kwargs):
        return self._fs.open_json(*args, **kwargs)

    def open_rtid(self, rtid: str):
        url = self.get_url(rtid)
        print(url)
        return self._fs.open_json(url)

    def get_url(self, rtid: str, fmt: str = "ld+json"):
        """Return URL toward a given reference table for a given format

        Parameters
        ----------
        rtid: str
            Name of the vocabulary table to retrieve. Eg: 'R01'
        fmt: str, default: "ld+json"
            Format of the NVS server response. Can be: "ld+json", "rdf+xml" or "text/turtle".

        Returns
        -------
        str
        """
        if fmt == "ld+json":
            fmt_ext = "?_profile=nvs&_mediatype=application/ld+json"
        elif fmt == "rdf+xml":
            fmt_ext = "?_profile=nvs&_mediatype=application/rdf+xml"
        elif fmt == "text/turtle":
            fmt_ext = "?_profile=nvs&_mediatype=text/turtle"
        else:
            raise ValueError(
                "Invalid format. Must be in: 'ld+json', 'rdf+xml' or 'text/turtle'."
            )
        url = "{}/{}/current/{}".format
        return url(self.nvs, rtid, fmt_ext)
