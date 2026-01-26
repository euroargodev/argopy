import inspect
import warnings

import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Any
from dataclasses import dataclass

from argopy.errors import NoDataLeft, OptionValueError
from argopy.utils.locals import Asset
from argopy.utils.format import ppliststr, urnparser
from argopy.utils.checkers import to_list
from argopy.stores.nvs import NVS, id2urn
from argopy.related.vocabulary.concept import ArgoReferenceValue


@dataclass(frozen=True)
class Props:
    """ ArgoReferenceTable property holder """

    slots = ('nvs', 'identifier', 'parameter', 'long_name', 'description', 'version', 'date', 'uri', '_Vocabulary2Parameter', '_df', '_d', '_keys')

    attrs = ('nvs', 'identifier', 'parameter', 'long_name', 'description', 'version', 'date', 'uri')
    """A subset of slots, to be publicly exposed"""

    keys = ('identifier', 'parameter', 'long_name', 'description', 'version', 'date', 'uri')
    """A subset of attrs, to be used in export methods (columns/keys selection)"""


class ArgoReferenceTable:
    """A class to work with one Argo reference table

    For instance the vocabulary for "Argo sensor models", corresponds to the Argo reference table 27 ("R27")
    and is used to document possible values of the "SENSOR_MODEL" parameter in netcdf files.

    An Argo reference table is a NVS "vocabulary", aka a SKOS "collection".

    .. note::
        ArgoNVSReferenceTables.tbl('R25')  # Deprecated API
        ArgoReferenceTable('R25').to_dataframe()  # New API, not backward compatible (new column names)

    Examples
    --------
    .. code-block:: python
        :caption: Creation and Attributes

        from argopy import ArgoReferenceTable

        # Use an Argo parameter name, documented by one of the Argo reference tables:
        art = ArgoReferenceTable('SENSOR')

        # or a reference table identifier:
        art = ArgoReferenceTable('R25')

        # or a URN:
        art = ArgoReferenceTable.from_urn('SDN:R25::CTD_TEMP')

        # All possible attributes are listed in:
        art.attrs

        # Reference Table attributes:
        art.parameter   # Name of the netcdf dataset parameter filled with values from this table
        art.identifier  # Reference Table ID
        art.description # [nvs['@graph']['@type']=='skos:Collection']["dc:description"]
        art.uri         # [nvs['@graph']['@type']=='skos:Collection']["@id"]
        art.version     # [nvs['@graph']['@type']=='skos:Collection']['owl:versionInfo']
        art.date        # [nvs['@graph']['@type']=='skos:Collection']['dc:date']

        # Raw NVS json data:
        art.nvs

    .. code-block:: python
        :caption: Indexing and values

        from argopy import ArgoReferenceTable
        art = ArgoReferenceTable('SENSOR')

        # Values (or concept) within this reference table:
        len(art)     # Number of reference values
        art.keys()   # List of reference values name
        art.values() # List of :class:`ArgoReferenceValue`

        # Check for values:
        'CTD_TEMP_CNDC' in art  # Return True

        # Index by value key, like a simple dictionary:
        art['CTD_TEMP_CNDC']  # Return a :class:`ArgoReferenceValue` instance

        # Allows to iterate over all values/concepts:
        for concept in art:
        	print(concept.name, concept.urn)

    .. code-block:: python
        :caption: Export methods

        from argopy import ArgoReferenceTable
        art = ArgoReferenceTable('SENSOR')

        # Export table content to a pd.DataFrame:
        art.to_dataframe()
        art.to_dataframe(columns=['name', 'deprecated'])  # Select attributes to export in columns

        # Export to json structure:
        # (basic export of NVS data)
        art.to_json()  # In memory
        art.to_json('referance_table.json')  # To a json file

        # Export to a dictionary:
        art.to_dict()
        art.to_dict(keys=['name', 'deprecated'])  # Select attributes to export in dictionary keys

    .. code-block:: python
        :caption: Search the table

        # Search methods (return a list of :class:`ArgoReferenceValue` with match):
        # Any of the :class:`ArgoReferenceValue` attribute can be searched
        art.search(name='RAMSES')         # Search in values name
        art.search(definition='imaging')  # Search in values definition
        art.search(long_name='TriOS')     # Search in values long name

        # Possible change to output format:
        art.search(deprecated=True, output='df')  # To a :class:`pd.DataFrame`

    """
    __slots__ = Props.slots

    attrs : tuple[str] = Props.attrs
    """Public attributes"""

    def __init__(self, identifier: str | None = None, *args, **kwargs) -> None:
        # Internal placeholders:
        self._Vocabulary2Parameter : dict[str, str] = Asset.load('vocabulary:mapping')['data']['Vocabulary2Parameter']
        self._df : pd.DataFrame | None = None  # Dataframe export
        self._d : dict[str, ArgoReferenceValue] | None = {}  # Dictionary of ArgoReferenceValue for all table concept

        if identifier in self._Vocabulary2Parameter:
            self.identifier : str = identifier
            self.parameter : str = self._Vocabulary2Parameter[identifier]
        elif identifier in self._Vocabulary2Parameter.values():
            self.parameter : str = identifier
            self.identifier : str = [k for k, v in self._Vocabulary2Parameter.items() if v == identifier][0]
        else:
            raise ValueError(f"Unknown Reference Table '{identifier}'. Possible values are: \nIDs like: {ppliststr([k for k in self._Vocabulary2Parameter], last='or')}\nNames like: {ppliststr([k for k in self._Vocabulary2Parameter.values()], last='or')}")

        # Once we have an id in 'name' we can load raw data from NVS
        self.nvs : dict[str, Any] = NVS().load_vocabulary(self.identifier)

        # And populate all attributes:
        Collection : dict[str, str] = [item for item in self.nvs['@graph'] if item['@type'] == 'skos:Collection'][0]
        """The NVS skos collection for this vocabulary"""

        self.long_name : str = Collection['skos:prefLabel']
        self.description : str = Collection["dc:description"]
        self.version : str = Collection['owl:versionInfo']
        self.date : pd.Timestamp = pd.to_datetime(Collection['dc:date'])
        self.uri : str = Collection["@id"]

        # Retrieve the list of concept names
        """
        Two methods:
        1- From the skos:Collection list of members:
            >>> values = [m['@id'].split("/")[-2] for m in Collection['skos:member']]
        2- From skos:Concept in the @graph:
            >>> values = [c['skos:altLabel'] for c in [item for item in self['@graph'] if item['@type'] == 'skos:Concept']]
        We stick to Collection for consistency with other attributes gathering
        """
        self._keys : list[str] = [m['@id'].split("/")[-2] for m in Collection['skos:member']]
        """List of this Reference Table value names, aka list of Concept names"""
        self._keys.sort()

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only policy after instantiation for public attributes"""
        if attr in self.__slots__ and attr[0] != '_' and inspect.stack()[1][3] != '__init__':
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        ArgoReferenceTable.__dict__[attr].__set__(self, value)

    def keys(self):
        return self._keys

    def values(self):
        return [self[v] for v in self.keys()]

    def __repr__(self):
        summary = [f"<argo.reference.table> '{self.identifier}'/'{self.parameter}'"]
        summary.append(f'long_name: "{self.long_name}"')
        summary.append(f'description: "{self.description}"')
        summary.append(f"uri: {self.uri}")
        summary.append(f"version: {self.version} ({self.date})")
        summary.append(f"keys[{len(self)}]: {ppliststr(self.keys(), n=10)}")
        return "\n".join(summary)

    def __str__(self):
        return f"ArgoReferenceTable('{self.parameter}')"

    def __contains__(self, item):
        return item in self.keys()

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for v in self.keys():
            yield self[v]

    def __getitem__(self, key : str):
        """Get a :class:`ArgoReferenceValue` instance from table key

        Reference Values are internally stored in a dictionary.
        In a lazy approach, only Reference Values reached with this '__getitem__' progressively populates the dictionary, which is empty at instantiation.
        Dictionary value is returned if a Reference Value has already been reached.
        """
        if not isinstance(key, str):
            raise TypeError("Indexing is only possible with a string index")
        ref_value : str | None = None
        if key in self.keys():
            ref_value = key
        if ref_value is not None:
            if self._d.get(ref_value, None) is None:
                """
                The naive method to call here is ArgoReferenceValue(name, reference):
                >>> self._d.update({ref_value: ArgoReferenceValue(ref_value, reference=self.identifier)})
                But this is could be very slow for large reference tables because it triggers one NVS fetch for
                each concept.
                Hopefully this naive method is not necessary since all concepts data are already in `self.nvs`:
                """
                data : list[dict] = [item for item in self.nvs['@graph'] if item['@type'] == 'skos:Concept' and item['skos:altLabel'] == ref_value]
                if len(data) == 1:
                    self._d.update({ref_value: ArgoReferenceValue.from_dict(data=data[0])})
                else:
                    # Temporary fix for https://github.com/OneArgo/ArgoVocabs/issues/186:
                    data = [item for item in self.nvs['@graph'] if
                            item['@type'] == 'skos:Concept' and urnparser(id2urn(item['@id']))['termid'] == ref_value]
                    self._d.update({ref_value: ArgoReferenceValue.from_dict(data=data[0])})
            return self._d[ref_value]
        raise ValueError(f"Invalid reference value '{key}'")

    def _ipython_key_completions_(self):
        """Provide method for key-autocompletions in IPython."""
        return [p for p in self.keys()]

    @classmethod
    def from_urn(cls, urn : str) -> 'ArgoReferenceTable':
        urn = urnparser(urn)
        return cls(urn['listid'])

    def to_dataframe(self, columns : list[str] | None = None) -> DataFrame | None:
        """Export all reference values attributes to a :class:`pd.DataFrame`

        Default column names are given by the :attr:`ArgoReferenceValue.keys` attribute.

        Parameters
        ----------
        columns: list[str] | None, optional, default=None
            Column names to insert into the output. By default, None, will include all available :attr:`ArgoReferenceValue.keys` attributes.

        Returns
        -------
        :class:`pd.DataFrame`
        """
        """
        Also note that we could create a dataframe directly from self.nvs json data
        But by design, we want to stick to using keys return by ArgoReferenceValue attributes,
        so that there is only one place determining how to map nvs json jargon onto a user-friendly facade,
        and that is the ArgoReferenceValue class.
        """
        if columns is None:
            cols = ArgoReferenceValue.keys
        else:
            cols = []
            for c in to_list(columns):
                if c not in ArgoReferenceValue.keys:
                    raise OptionValueError(
                        f"Invalid columns name '{c}'. Valid values are: {ppliststr(ArgoReferenceValue.keys)}")
                cols.append(c)
            if len(cols) == 0:
                raise OptionValueError(
                    f"No valid column names in '{ppliststr(columns)}'. Valid values are: {ppliststr(ArgoReferenceValue.keys, last='or')}")

        def todf(columns: list[str]):
            dict_list = []
            for value in self:
                d = value.to_dict()
                d = {key: d[key] for key in columns}
                dict_list.append(d)
            return pd.DataFrame(dict_list)

        if self._df is None:
            self._df = todf(cols)
        elif set(cols) != set(self._df.columns.tolist()):
            self._df = todf(cols)
        return self._df

    def search(self, **kwargs) -> list[ArgoReferenceValue] | pd.DataFrame:
        """Search in table list of :class:`ArgoReferenceValue` attributes

        Parameters
        ----------
        tuple(str, str)
            Attributes to search among :attr:`ArgoReferenceValue.keys`.

            Use the specific argument `output='df'` to return a :class:`pd.DataFrame`.

        Returns
        -------
        list[ArgoReferenceValue] | :class:`pd.DataFrame`

        Raises
        ------
        :class:`NoDataLeft`

        Examples
        --------
        .. code:: python

            from argopy import ArgoReferenceTable
            art = ArgoReferenceTable('SENSOR')

            # Search in specific attributes:
            art.search(name='RAMSES')
            art.search(definition='imaging')
            art.search(long_name='TriOS')

            # Search in more than one attribute:
            art.search(name='RAMSES', deprecated=False)

            # Control output format:
            art.search(name='CTD', output='df')

        """
        # Note that we could implement search on self.nvs json data
        # But we want to stick to using keys from ArgoReferenceValue attributes
        # so that there is only place determining how to map nvs json jargon onto user-friendly facade
        # i.e. the ArgoReferenceValue.

        # Get output format:
        output = None
        if kwargs.get('output', None) is not None:
            output = kwargs.get('output')
            kwargs.pop('output')

        # Search key validation:
        keys = [key for key in kwargs if key in ArgoReferenceValue.keys]

        # Search using the dataframe view of this reference table:
        df = self.to_dataframe()
        filters = []
        for key in keys:
            if df[key].dtype in ['str', 'object']:
                filters.append(df[key].str.contains(str(kwargs[key]), regex=True, case=False))
            elif df[key].dtype == 'datetime64[ns]':
                warnings.warn("No search method implemented for datetime")
            elif df[key].dtype == 'bool':
                filters.append(df[key] == kwargs[key])
        mask = np.logical_and.reduce(filters)
        df = df[mask]
        if df.shape[0] > 0:
            if output is None:
                return [self[name] for name in df['name'].tolist()]
            else:
                return df.reset_index(drop=True)
        else:
            raise NoDataLeft("This search return no data")

    def to_dict(self, keys : list[str] | None = None) -> dict[str, Any]:
        """Export reference table attributes to a dictionary"""
        if keys is None:
            validated_keys = Props.keys
        else:
            validated_keys = []
            for k in to_list(keys):
                if k not in Props.keys:
                    raise OptionValueError(
                        f"Invalid keys name '{k}'. Valid values are: {ppliststr(Props.keys)}")
                validated_keys.append(k)
            if len(validated_keys) == 0:
                raise OptionValueError(
                    f"No valid key names in '{ppliststr(keys)}'. Valid values are: {ppliststr(Props.keys, last='or')}")

        d = {}
        for key in validated_keys:
            d.update({key: getattr(self, key)})
        return d

    def to_json(self, *args, **kwargs):
        raise NotImplementedError('Coming up soon !')
