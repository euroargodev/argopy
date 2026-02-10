from abc import ABC, abstractmethod
from typing import Callable
from argopy.utils.locals import caller_function


class NVSProto(ABC):
    online : bool | None = None
    """Flag indication if we get data from NVS server (True) or static assets (False)"""

    uid: str = None
    """Unique NVS store instance ID"""

    def __setattr__(self, attr, value):
        """Set attribute value, with a 'read-only after instantiation' policy for public attributes"""
        if attr in [key for key in self.__dir__() if key[0] != "_"] and not caller_function().startswith("__init"):
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

    @abstractmethod
    def load_vocabulary(self, rtid: str, fmt: str = "json") -> dict:
        """Load a NVS vocabulary, i.e. a SKOS collection, as a :class:`dict`

        Parameters
        ----------
        rtid: str
            Name of the vocabulary, i.e. SKOS collection, to retrieve. Eg: 'R01'
        fmt: str, default: "ld+json"
            Format of the NVS server response. Can be: "json", "xml" or "turtle".

        Returns
        -------
        dict

        Notes
        -----
        When running offline, only the 'json' format is available.
        """
        raise NotImplementedError

    @abstractmethod
    def load_concept(self, conceptid: str, rtid: str | None = None, fmt: str = "json") -> dict:
        """Load a NVS concept, i.e. a SKOS concept, as a :class:`dict`

        Parameters
        ----------
        conceptid: str
            Name of the concept, SKOS concept, to retrieve. Eg: 'AANDERAA_OPTODE_3835'
        rtid: str, optional, default = None
            Name of the vocabulary, SKOS collection, for this concept. Eg: 'R27'.
            If set to None, we try to guess it, but if the concept is not found or can be found in more than one vocabulary, an error is raise.
        fmt: str, default: "json"
            Format of the NVS server response. Can be: "json", "xml" or "turtle".

        Returns
        -------
        dict

        Notes
        -----
        When running offline, only the 'json' format is available.
        """
        raise NotImplementedError
