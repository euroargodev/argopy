from typing import Any

from argopy.utils.locals import Asset
from argopy.stores.nvs.spec import NVSProto
from argopy.stores.nvs.utils import concept2vocabulary


def fmt2uri(fmt):
    d = {
        "json": "",
        # "xml": None,
        # "turtle": None,
    }

    if fmt in d.keys():
        return d[fmt]

    raise ValueError(
            "Invalid format. Must be 'json' in offline mode."
        )



class NVS(NVSProto):
    online = False

    nvs: str = None
    """Url to NVS"""

    _instance: "NVS | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "NVS":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        if not self._initialized:
            self.nvs = "<local.static.assets>"
            self._initialized = True
        self.uid = id(self)

    def _vocabulary2uri(self, rtid: str, fmt: str = "json"):
        """Return URI of a given vocabulary

        Parameters
        ----------
        rtid: str
            Name of the vocabulary (SKOS collection) to address. Eg: 'R01'.

        Returns
        -------
        str
        """
        return f"vocabulary:offline:{rtid}{fmt2uri(fmt)}"

    def load_vocabulary(self, rtid: str, fmt: str = "json") -> dict:
        url = self._vocabulary2uri(rtid, fmt=fmt)
        return Asset().load(url)['data']

    def _concept2uri(self, conceptid: str, rtid: str | None = None, fmt: str = "json") -> str:
        """Return URI of a given concept

        Parameters
        ----------
        conceptid: str
            Name of the concept (SKOS concept) to retrieve. Eg: 'AANDERAA_OPTODE_3835'.
        rtid: str, optional, default = None
            Name of the vocabulary (SKOS collection) for this concept. Eg: 'R27'.
            If set to None, we try to guess it, but if the concept is not found or can be found in more than one vocabulary, an error is raised.

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
        return f"vocabulary:offline:{rtid}:{conceptid}{fmt2uri(fmt)}"

    def load_concept(self, conceptid: str, rtid: str | None = None, fmt: str = "json") -> dict:
        url = self._concept2uri(conceptid, rtid, fmt=fmt)
        return Asset().load(url)['data']

    def load_mapping(self, subjectid: str, objectid: str, fmt: str = "json") -> dict:
        raise NotImplementedError()
