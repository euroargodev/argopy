from typing import Any
from functools import lru_cache
import inspect

from argopy.options import OPTIONS
from argopy.stores.implementations.http import httpstore
from argopy.stores.nvs.spec import NVSProto
from argopy.stores.nvs.utils import concept2vocabulary


class NVS(NVSProto):
    online = True

    nvs: str = None
    """Url to the NVS"""

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
            self.nvs = kwargs.get("nvs", OPTIONS["nvs"])
            # self._vocabulary = self._ls_vocabulary()
            self._initialized = True
        self.uid = id(self)


    def __setattr__(self, attr, value):
        """Set attribute value, with read-only after instantiation policy for public attributes"""
        if (
            attr in [key for key in self.__dir__() if key[0] != "_"]
            and inspect.stack()[1][3] != "__init__"
        ):
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        self.__dict__[f"{attr}"] = value

    @lru_cache
    def open_json(self, *args, **kwargs):
        return self._fs.open_json(*args, **kwargs)

    def vocabulary2url(self, rtid: str, fmt: str = "ld+json"):
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

    def concept2url(self, conceptid: str, rtid: str | None = None, fmt: str = "ld+json"):
        """Return URL toward a given concept for a given format

        Parameters
        ----------
        conceptid: str
            Name of the vocabulary concept to retrieve. Eg: 'R01'
        rtid: str
            Name of the vocabulary for this concept. Eg: 'R01'
        fmt: str, default: "ld+json"
            Format of the NVS server response. Can be: "ld+json", "rdf+xml" or "text/turtle".

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
        url = "{}/{}/current/{}/{}".format
        return url(self.nvs, rtid, conceptid, fmt_ext)

    @lru_cache
    def load_vocabulary(self, rtid: str) -> dict:
        url = self.vocabulary2url(rtid)
        return self._fs.open_json(url)

    @lru_cache
    def load_concept(self, conceptid: str, rtid: str | None = None) -> dict:
        url = self.concept2url(conceptid, rtid)
        return self._fs.open_json(url)

    # def _ls_vocabulary(self):
    #     data = self._fs.open_json(f'{self.nvs}/?_profile=nvs&_mediatype=application/ld+json')
    #
    #     def is_admt(item):
    #         return item['dc:creator'] == 'Argo Data Management Team'
    #
    #     id_list = [item for item in data['@graph'] if is_admt(item)]
    #
    #     valid_ref = []
    #     for item in id_list:
    #         valid_ref.append(item['@id'].replace(f"{self.nvs}/", "").replace("/current/", ""))
    #     #     valid_ref.append({
    #     #         'id': item['@id'].replace("http://vocab.nerc.ac.uk/collection/", "").replace("/current/", ""),
    #     #         'altLabel': item['skos:altLabel'],
    #     #         'prefLabel': item['skos:prefLabel'],
    #     #         'description': item['dc:description'],
    #     #         'date': item['dc:date'],
    #     #         'uri': item['@id'],
    #     #     })
    #     # df = pd.DataFrame(valid_ref).sort_values('id', axis=0).reset_index(drop=1)
    #     return valid_ref
