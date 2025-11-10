import inspect
import pandas as pd
from typing import Any, TypeAlias
from os import PathLike
from pathlib import Path

from ...errors import OptionValueError
from ...utils import urnparser, Asset, ppliststr, to_list, Encoder
from .nvs import concept2vocabulary, check_vocabulary, id2urn, NVS


FilePath: TypeAlias = str | PathLike[str]


class Props:
    """ :class:`ArgoReferenceValue` property holder

    This should allow to make the difference between the class logic/attributes and the meta-data to expose.
    """

    slots = ('name', 'reference', 'long_name', 'definition', 'deprecated', 'version', 'date', 'uri', 'urn', 'parameter', 'related', 'broader', 'narrower', '_nvs', '_context', '_from')
    """All possible class attributes"""

    attrs = ('nvs', 'name', 'reference', 'long_name', 'definition', 'deprecated', 'version', 'date', 'uri', 'urn', 'parameter', 'related', 'broader', 'narrower', 'context')
    """Attributes to be publicly exposed (and are read-only)"""

    cols = ('name', 'reference', 'long_name', 'definition', 'deprecated', 'version', 'date', 'uri', 'urn', 'parameter', 'related', 'broader', 'narrower', 'context')
    """Attributes to be used to validate export/search possible values"""


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
        :caption: Creation and Attributes

        from argopy import ArgoReferenceValue

        # One possible value for the Argo parameter 'SENSOR_MODEL':
        val = ArgoReferenceValue('AANDERAA_OPTODE_3835')

        # For ambiguous value seen in more than one Reference Table
        val = ArgoReferenceValue('4', reference='RT_QC_FLAG')
        val = ArgoReferenceValue('4', reference='RR2')

        # From NVS/URN jargon:
        val = ArgoReferenceValue.from_urn('SDN:R27::AANDERAA_OPTODE_3835')


    .. code-block:: python
        :caption: Attributes

        # All possible attributes are listed in:
        val.attrs

        # Reference Value attributes:
        val.name       # Term-id of the URN, eg 'AANDERAA_OPTODE_3835'
        val.long_name  # nvs["skos:prefLabel"]["@value"]
        val.definition # nvs["skos:definition"]["@value"]
        val.deprecated # nvs["owl:deprecated"]
        val.parameter  # The netcdf parameter this concept applies to (eg 'SENSOR_MODEL')
        val.reference  # The reference table this concept belongs to, can be used on a ArgoReferenceTable (eg 'R27')

        # Other reference Value attributes (more technical):
        val.version    # nvs["owl:versionInfo"]
        val.date       # nvs["dc:date"]
        val.uri        # nvs["@id"]
        val.urn        # nvs["skos:notation"]

        # Relationships with other Reference Values or Context:
        val.related    # nvs["skos:related"]
        val.broader    # nvs["skos:broader"]
        val.narrower   # nvs["skos:narrower"]
        val.context    # nvs["@context"]

        # Raw NVS json data:
        val.nvs

    .. code-block:: python
        :caption: Indexing

        from argopy import ArgoReferenceValue

        # Load some reference value:
        val = ArgoReferenceValue('AANDERAA_OPTODE_3835')

        # All attributes can be obtained with name indexing, like a simple dictionary:
        val['definition']

    .. code-block:: python
        :caption: Export methods

        from argopy import ArgoReferenceValue

        # Load some reference value:
        val = ArgoReferenceValue('AANDERAA_OPTODE_3835')

        # Export to a dictionary:
        val.to_dict()
        val.to_dict(keys=['name', 'deprecated'])  # Select attributes to export in dictionary keys

        # Export to json structure:
        # (basic export of NVS data)
        val.to_json()  # In memory
        val.to_json('reference_value.json')  # To a json file

        # Export relationships with other concept as :class:`pd.DataFrame`
        val.to_mapping(predicate=['related'])
        val.to_mapping(predicate=['broader', 'narrower'])
        val.to_mapping(predicate=['context'])

    """
    __slots__ = Props.slots

    attrs : tuple[str] = Props.attrs
    """Public attributes"""

    def __init_implicit(self, name: str | None = None, reference: str | None = None, **kwargs) -> None:
        self._from = 'nvs'
        self.name = name
        reftable = concept2vocabulary(name)  # Return IDs
        if reftable is None:
            raise ValueError('Invalid Reference Value')
        if reference is not None:
            reference = check_vocabulary(reference)  # Return a table ID, whatever the input
            if reference not in reftable:
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
        self._nvs = NVS().load_concept(self.name, self.reference)

    def __init_explicit(self, data: Any) -> None:
        self._from = 'json'
        self._nvs = data
        self.name = self.nvs['skos:altLabel']
        if self.name == '' or self.name is None:
            self.name = urnparser(id2urn(self.nvs['@id']))['termid']
        self.reference = urnparser(self.nvs['dce:identifier'])['listid'] # eg 'dce:identifier' = 'SDN:R27::UNKNOWN'

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
        self._context = self.nvs.get('@context', None)

        # todo: support mapping (https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#ivb-mappings)
        # Relation can be:
        # "narrower/broader" when there is a hierarchy between the subject and the object
        # "related" when the subject is related to the object without strict hierarchy
        # Eg: 'AANDERAA_OPTODE_3830' concept:
        #  'skos:related': {'@id': 'http://vocab.nerc.ac.uk/collection/R25/current/OPTODE_DOXY/'},
        #  'skos:broader': {'@id': 'http://vocab.nerc.ac.uk/collection/R26/current/AANDERAA/'},
        self.related = None
        if self.nvs.get('skos:related', None) is not None:
            self.related = to_list(self.nvs.get('skos:related', None))
        self.broader = None
        if self.nvs.get('skos:broader', None) is not None:
            self.broader = to_list(self.nvs.get('skos:broader', None))
        self.narrower = None
        if self.nvs.get('skos:narrower', None) is not None:
            self.narrower = to_list(self.nvs.get('skos:narrower', None))

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only after instantiation policy for public attributes"""
        if attr in self.attrs and not inspect.stack()[1][3].startswith('__init'):
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        ArgoReferenceValue.__dict__[attr].__set__(self, value)

    def __repr__(self):
        summary = [f"<argo.reference.value><{self.parameter}.{self.name}>"]
        summary.append(f'long_name: "{self.long_name}"')
        summary.append(f"version: {self.version} ({self.date})")
        summary.append(f"uri: {self.uri}")
        summary.append(f'definition: "{self.definition}"')
        summary.append(f'urn: "{self.urn}"')
        summary.append(f'deprecated: {"True" if self.deprecated else "False"}')
        summary.append(f"reference/parameter: {self.reference}/{self.parameter}")

        for relation in ['related', 'broader', 'narrower']:
            if getattr(self, relation, None) is not None:
                # list of items like: {'@id': 'http://vocab.nerc.ac.uk/collection/R23/current/PROVOR_II/'}
                rels = getattr(self, relation)
                # Format the list as a list of items like 'R23/PROVOR_II':
                urns = [urnparser(id2urn(r['@id'])) for r in rels]
                urns = [f"{u['listid']}/{u['termid']}" for u in urns]
                # Final print:
                summary.append(f"{relation}[{len(urns)}]: {ppliststr(urns)}")
            else:
                summary.append(f"{relation}: n/a")

        if getattr(self, 'context', None) is not None:
            keys = list(self.context.keys())
            summary.append(f"context[{len(keys)}]: {ppliststr(keys)}")
        else:
            summary.append(f"context: {'(not loaded yet, use key indexing to load)' if self._from == 'json' else 'n/a'}")
        return "\n".join(summary)

    def __getitem__(self, key):
        if key == 'context':
            """ 'context' requires a special treatment because this is the only attribute that is not filled
            when the ArgoReferenceValue instance is created using json data from a Reference Table graph 
            concept and the from_dict method, typically in this case:
            >>> val = ArgoReferenceTable('PLATFORM_FAMILY')['FLOAT_COASTAL']
            This 'val' instance has no 'context' attribute.            
            So, when we call on "val['context']" we need to trigger full NVS data fetching of the concept, which also update the internal nvs object.            
            """
            if self.nvs.get('@context', None) is None:
                # Update NVS data:
                self._nvs : dict[str, str] = NVS().load_concept(self.name, self.reference)
                # Fill in context attribute:
                self._context : str | None = self.nvs.get('@context', None)
            return getattr(self, 'context')
        elif key in self.__slots__:
            return getattr(self, key)
        raise ValueError(f"Unknown attribute '{key}'")

    @property
    def nvs(self):
        return self._nvs

    @property
    def context(self):
        return self._context

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

    def to_dict(self, columns : list[str] | None = None) -> dict[str, Any]:
        """Export reference value attributes to a dictionary"""
        if columns is None:
            cols = Props.cols
        else:
            cols = []
            for c in to_list(columns):
                if c not in Props.cols:
                    raise OptionValueError(
                        f"Invalid columns name '{c}'. Valid values are: {ppliststr(Props.cols)}")
                cols.append(c)
            if len(cols) == 0:
                raise OptionValueError(
                    f"No valid column names in '{ppliststr(columns)}'. Valid values are: {ppliststr(Props.cols, last='or')}")

        d = {}
        for key in cols:
            d.update({key: getattr(self, key)})
        return d

    def to_json(self, path: FilePath | None = None, **kwargs):
        """Export to a JSON string or path

        Parameters
        ----------
        path: str, path object, file-like object, or None, default None
            String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function. If None, the result is returned as a string.
        **kwargs
            All other arguments are passed to :class:`json.dumps` or :class:`json.dump`

        Returns
        -------
        None or str
            If path is None, returns the resulting json format as a string. Otherwise returns None.
        """
        import json

        # Get data to export:
        data = self.to_dict()

        # Make sure we have an appropriate JSON encoder for pandas data types
        if kwargs.get('cls', None) is None:
            kwargs.update({'cls': Encoder})

        # Export:
        if path is None:
            return json.dumps(data, **kwargs)
        else:
            if getattr(path, 'write', None) is None:
                with open(Path(path), 'w') as fp:
                    return json.dump(data, fp, **kwargs)
            else:
                return json.dump(data, path, **kwargs)

    def to_mapping(self):
        raise NotImplementedError