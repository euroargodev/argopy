import inspect
import pandas as pd
from functools import lru_cache
from typing import Any

from ...utils import urnparser, Asset, ppliststr
from .nvs import concept2vocabulary, NVS


class ArgoReferenceValue:
    """A class to work with an Argo Reference Value, i.e. a NVS vocabulary "concept"

    An Argo Reference Value is one possible and documented value for one Argo parameter.

    For instance, 'AANDERAA_OPTODE_3835' is an Argo Reference Value for the 'SENSOR_MODEL' parameter.

    Notes
    -----
    The :class:`ArgoReferenceValue` class is low-level interface and can typically be ignored. However, the word “ArgoReferenceValue” appears often enough in the code and documentation that is useful to understand.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoReferenceValue

        avc = ArgoReferenceValue('AANDERAA_OPTODE_3835')  # One possible value for the Argo parameter 'SENSOR_MODEL'
        avc = ArgoReferenceValue.from_urn('SDN:R27::AANDERAA_OPTODE_3835')

        avc = ArgoReferenceValue('4', reference='RR2')  # For ambiguous value seen in more than one Reference Table

        # Reference Value attributes:
        avc.name
        avc.long_name  # data["skos:prefLabel"]["@value"]
        avc.definition # data["skos:definition"]["@value"]
        avc.deprecated # data["owl:deprecated"]
        avc.version    # data["owl:versionInfo"]
        avc.date       # data["dc:date"]
        avc.uri        # data["@id"]
        avc.urn        # data["skos:notation"]
        avc.context    # data["@context"]
        avc.nvs        # Raw NVS json data

        avc.parameter  # The netcdf parameter this concept applies to (eg 'SENSOR_MODEL')
        avc.reference  # The reference table this concept belongs to, can be used on a ArgoReferenceTable (eg 'R27')

        # Attributes can also be obtained with indexing:
        avc['definition']

    """
    __slots__ = ['nvs', 'name', 'reference', 'long_name', 'definition', 'deprecated', 'version', 'date', 'uri', 'urn', 'parameter', 'context']

    def __init_implicit(self, name: str | None = None, reference: str | None = None, **kwargs) -> None:
        self.name = name
        reftable = concept2vocabulary(name)
        if reftable is None:
            raise ValueError('Invalid Reference Value')
        if reference is not None and reference not in reftable:
            raise ValueError(
                f"Reference Table '{reference}' not valid for the '{name}' Reference Value, should be one in: {reftable}")
        if reference is None:
            if len(reftable) > 1:
                raise ValueError(
                    f"This Reference Value appears in more than one Reference Table: {reftable}. You must specified with the 'reference' argument which one to use.")
            else:
                self.reference = reftable[0]
        else:
            self.reference = reference  # eg 'R27'

        # Once we have a 'name' and a 'reference', we can load raw data from NVS
        self.nvs = NVS().load_concept(self.name, self.reference)

    def __init_explicit(self, data: Any) -> None:
        self.nvs = data
        self.name = self.nvs['skos:altLabel']
        self.reference = urnparser(self.nvs['dce:identifier'])['listid'] # eg 'SDN:R27::UNKNOWN'

    def __init__(self, name: str, reference: str | None = None, **kwargs) -> None:
        if kwargs.get('data', None) is None:
            self.__init_implicit(name=name, reference=reference)
        else:
            self.__init_explicit(data=kwargs.get('data'))

        # And populate all attributes:
        self.long_name = self.nvs["skos:prefLabel"]["@value"]
        self.definition = self.nvs["skos:definition"]["@value"]
        self.deprecated = True if self.nvs["owl:deprecated"] == 'True' else False
        self.version = self.nvs['owl:versionInfo']
        self.date = pd.to_datetime(self.nvs['dc:date'])
        self.uri = self.nvs["@id"]
        self.urn = self.nvs["skos:notation"]
        self.parameter = Asset().load('vocabulary:mapping')['data']['Vocabulary2Parameter'][self.reference]
        self.context = self.nvs.get('@context', None)

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only after instantiation policy for public attributes"""
        if attr in self.__slots__ and not inspect.stack()[1][3].startswith('__init'):
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        ArgoReferenceValue.__dict__[attr].__set__(self, value)

    def __repr__(self):
        summary = [f"<argo.reference.value><{self.parameter}.{self.name}>"]
        summary.append(f'long_name: "{self.long_name}"')
        summary.append(f"version: {self.version} ({self.date})")
        summary.append(f"uri: {self.uri}")
        summary.append(f'definition: "{self.definition}"')
        summary.append(f"urn: {self.urn}")
        summary.append(f"reference: table {self.reference}")
        summary.append(f'deprecated: {"True" if self.deprecated else "False"}')
        if getattr(self, 'context', None) is not None:
            keys = list(self.context.keys())
            summary.append(f"context[{len(keys)}]: {ppliststr(keys)}")
        else:
            summary.append(f"context: n/a")
        return "\n".join(summary)

    def __getitem__(self, key):
        if key == 'context':
            if self.nvs.get('@context', None) is None:
                self.nvs = NVS().load_concept(self.name, self.reference)
                self.context = self.nvs.get('@context', None)
            return getattr(self, 'context')
        elif key in self.__slots__:
            return getattr(self, key)
        raise ValueError(f"Unknown attribute '{key}'")

    @classmethod
    def from_urn(cls, urn: str = None) -> 'ArgoReferenceValue':
        urn = urnparser(urn)
        return cls(urn['termid'], reference=urn['listid'])

    @classmethod
    def from_dict(cls, data: dict = None) -> 'ArgoReferenceValue':
        """Create a :class:`ArgoReferenceValue` from a dictionary (JSON-like)

        Examples
        --------
        .. code-block :: python
            :caption: Expected dictionary structure

            {'@id': 'http://vocab.nerc.ac.uk/collection/R27/current/UNKNOWN/',
             'pav:authoredOn': '2019-10-11 14:49:00.0',
             'pav:hasCurrentVersion': {'@id': 'http://vocab.nerc.ac.uk/collection/R27/current/UNKNOWN/1/'},
             'dce:identifier': 'SDN:R27::UNKNOWN',
             'pav:version': '1',
             'skos:notation': 'SDN:R27::UNKNOWN',
             'skos:altLabel': 'UNKNOWN',
             'dc:date': '2019-10-11 14:49:00.0',
             'owl:versionInfo': '1',
             'skos:prefLabel': {'@language': 'en', '@value': 'Unknown sensor model'},
             'dc:identifier': 'SDN:R27::UNKNOWN',
             'skos:note': {'@language': 'en', '@value': 'accepted'},
             'owl:deprecated': 'false',
             'void:inDataset': {'@id': 'http://vocab.nerc.ac.uk/.well-known/void'},
             'skos:definition': {'@language': 'en', '@value': 'Sensor model is unknown.'},
             '@type': 'skos:Concept'}
        """
        return cls('', data = data)

    @lru_cache(maxsize=256)
    def to_dict(self):
        """name, reference are used in argument to force cache, not sure obj is enough, to be checked"""
        keys = self.__slots__.copy()
        keys.sort()
        d = {}
        for key in keys:
            d.update({key: getattr(self, key)})
        d.pop('nvs', None)
        d.pop('context', None)
        return d

    @classmethod
    def attrs(cls):
        return [a for a in cls.__slots__.copy() if a != 'nvs']
