from typing import Any, Callable
import inspect

from ...stores import httpstore
from ...utils import urnparser, Asset


class ArgoReferenceValue:
    """A class to work with an Argo Reference Value, i.e. a NVS vocabulary "concept"

    An Argo Reference Value is one possible and documented value of one Argo parameter.

    Using the AVTT/NVS jargon, this is a vocabulary **concept**.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoReferenceValue

        avc = ArgoReferenceValue('AANDERAA_OPTODE_3835')  # One possible value for the Argo parameter 'SENSOR_MODEL'
        avc = ArgoReferenceValue.from_urn('SDN:R27::AANDERAA_OPTODE_3835')

        avc.name       # pd.DataFrame['altLabel'] > urnparser(data["@graph"]['skos:notation'])['termid']
        avc.long_name  # pd.DataFrame['prefLabel'] > data["@graph"]["skos:prefLabel"]["@value"]
        avc.definition # pd.DataFrame['definition'] > data["@graph"]["skos:definition"]["@value"]
        avc.deprecated # pd.DataFrame['deprecated'] > data["@graph"]["owl:deprecated"]
        avc.uri        # pd.DataFrame['id'] > data["@graph"]["@id"]
        avc.urn        # pd.DataFrame['urn'] > data["@graph"]["skos:notation"]
        av._data       # Raw NVS json data

        avc.parameter  # The netcdf parameter this concept applies to (eg 'SENSOR_MODEL')
        avc.reftable   # The reference table this concept belongs to, can be used on a ArgoReferenceTable (eg 'R27')

    """
    name: str = None
    """Name of this Reference Value (eg 'AANDERAA_OPTODE_3835')"""

    reference: str = None
    """Reference Table this concept belongs to (eg 'R25')"""

    _fs: Any = None
    _instance: 'ArgoReferenceValue | None' = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> 'ArgoReferenceValue':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str | None = None, *args, **kwargs) -> None:
        if not self._initialized:
            self._fs = httpstore(cache=True)
            self._Vocabulary2Concept = Asset.load('vocabulary:mapping')['data']['Vocabulary2Concept']
            self._initialized = True
        reftable = self._name2reference(name)
        if reftable is None:
            raise ValueError('Invalid Reference Value')
        if kwargs.get('reference', None) is not None and kwargs.get('reference') not in reftable:
            raise ValueError(
                f"Reference Table '{kwargs.get('reference')}' not valid for the '{name}' Reference Value, should be one in: {reftable}")
        if kwargs.get('reference', None) is None:
            if len(reftable) > 1:
                raise ValueError(
                    f"This Reference Value appears in more than one Reference Table: {reftable}. You must specified with the 'reference' argument which one to use.")
            else:
                self.reference = reftable[0]
        else:
            self.reference = kwargs.get('reference')

        self.name = name

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only after instantiation policy for public attributes"""
        if attr in [key for key in self.__dir__() if key[0] != '_'] and inspect.stack()[1][3] != '__init__':
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        self.__dict__[f"{attr}"] = value

    def __repr__(self):
        props = [key for key in self.__dir__() if key[0] != '_' and not isinstance(getattr(self, key), Callable)]
        props = sorted(props)
        props_str = [f"{prop}='{getattr(self, prop)}'" for prop in props]
        return f"ArgoReferenceValue({', '.join(props_str)})"

    def _name2reference(self, name: str):
        """Map a 'Reference Value' to a 'Reference Table'

        Based on the NVS Vocabulary-to-Concept mapping in assets
        """
        name = name.strip().upper()
        found = []
        for vocabulary in self._Vocabulary2Concept.keys():
            if name in self._Vocabulary2Concept[vocabulary]:
                found.append(vocabulary)
        if len(found) == 0:
            return None
        return found

    @classmethod
    def from_urn(cls, urn: str = None) -> 'ArgoReferenceValue':
        urn = urnparser(urn)
        return cls(urn['termid'], reference=urn['listid'])
