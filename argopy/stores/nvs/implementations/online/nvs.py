from typing import Any
from functools import lru_cache

from argopy.options import OPTIONS
from argopy.stores.implementations.http import httpstore
from argopy.stores.nvs.spec import NVSProto
from argopy.stores.nvs.utils import concept2vocabulary


def fmt2urlparams(fmt):
    d = {
        "json": "application/ld+json",
        "xml": "application/rdf+xml",
        "turtle": "text/turtle"}

    if fmt in d.keys():
        return f"?_profile=nvs&_mediatype={d[fmt]}"

    raise ValueError(
            "Invalid format. Must be in: 'json', 'xml' or 'turtle'."
        )


class NVS(NVSProto):
    online = True

    nvs: str = None
    """Url to NVS"""

    _fs: Any = None
    _instance: "NVS | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "NVS":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        if not self._initialized:
            self._fs : httpstore = httpstore(
                cache=kwargs.get("cache", True),
                cachedir=kwargs.get("cachedir", OPTIONS["cachedir"]),
                timeout=kwargs.get("timeout", OPTIONS["api_timeout"]),
            )
            self.nvs = kwargs.get("nvs", OPTIONS["nvs"])
            self._initialized = True
        self.uid = id(self)

    def _downloader(self, fmt: str = "json"):
        if 'json' not in fmt:
            return lambda x: self._fs.download_url(x).decode('utf-8')
        return self._fs.open_json

    @lru_cache
    def open_json(self, *args, **kwargs):
        return self._fs.open_json(*args, **kwargs)

    def _vocabulary2uri(self, rtid: str, fmt: str = "json") -> str:
        """Return URI of a given vocabulary with a given format

        Parameters
        ----------
        rtid: str
            Name of the vocabulary (SKOS collection) to address. Eg: 'R01'.
        fmt: str, default: "json"
            Format of the NVS server response. Can be: "json", "xml" or "turtle".

        Returns
        -------
        str
        """
        url = "{}/{}/current/{}".format
        return url(self.nvs, rtid, fmt2urlparams(fmt))

    @lru_cache
    def load_vocabulary(self, rtid: str, fmt: str = "json") -> dict | Any:
        url = self._vocabulary2uri(rtid, fmt=fmt)
        return self._downloader(fmt)(url)

    def _concept2uri(self, conceptid: str, rtid: str | None = None, fmt: str = "json") -> str:
        """Return URI of a given concept, with a given format

        Parameters
        ----------
        conceptid: str
            Name of the concept (SKOS concept) to retrieve. Eg: 'AANDERAA_OPTODE_3835'
        rtid: str, optional, default = None
            Name of the vocabulary (SKOS collection) for this concept. Eg: 'R27'.
            If set to None, we try to guess it, but if the concept is not found or can be found in more than one vocabulary, an error is raised.
        fmt: str, default: "json"
            Format of the NVS server response. Can be: "json", "xml" or "turtle".

        Returns
        -------
        str
        """
        if rtid is None:
            reftable = concept2vocabulary(conceptid)
            if reftable is None:
                raise ValueError('Invalid Concept')
            if len(reftable) > 1:
                raise ValueError(
                    f"This Concept appears in more than one Vocabulary: {reftable}. You must specified with the 'rtid' argument which one to use.")
            else:
                rtid = reftable[0]

        url = "{}/{}/current/{}/{}".format
        return url(self.nvs, rtid, conceptid, fmt2urlparams(fmt))

    @lru_cache
    def load_concept(self, conceptid: str, rtid: str | None = None, fmt: str = "json") -> dict | Any:
        url = self._concept2uri(conceptid, rtid, fmt=fmt)
        return self._downloader(fmt)(url)

