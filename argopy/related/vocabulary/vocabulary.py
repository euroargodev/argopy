import inspect
import pandas as pd
import numpy as np
from pandas import DataFrame

from ...errors import NoDataLeft, OptionValueError
from ...utils import Asset, ppliststr
from .nvs import NVS
from .concept import ArgoReferenceValue


ArgoReferenceValue_attributes = ArgoReferenceValue.attrs()


class ArgoReferenceTable:
    """A class to work with one Argo reference table, i.e. a NVS "vocabulary"

    For instance the vocabulary for "Argo sensor models", corresponds to the Argo reference table 27 ("R27")
    and is used to document possible values of the "SENSOR_MODEL" parameter in netcdf files.

    .. note::
        ArgoNVSReferenceTables.tbl('R25')  # Deprecated API
        ArgoReferenceTable('R25').to_dataframe()  # New API, new column names


    Examples
    --------
    .. code-block:: python
        :caption: Creation and Attributes

        from argopy import ArgoReferenceTable

        ref = ArgoReferenceTable('R25')     # Use Reference Table identifier
        ref = ArgoReferenceTable('SENSOR')  # Or use an Argo parameter documented by one of the Argo reference tables

        # Reference Table attributes:
        ref.identifier  # Reference Table ID
        ref.description # [data['@graph']['@type']=='skos:Collection']["dc:description"]
        ref.uri         # [data['@graph']['@type']=='skos:Collection']["@id"]
        ref.version     # [data['@graph']['@type']=='skos:Collection']['owl:versionInfo']
        ref.date        # [data['@graph']['@type']=='skos:Collection']['dc:date']
        ref.nvs         # Raw NVS json data

        # Name of the netcdf dataset parameter filled with values from this table:
        ref.parameter  # eg 'SENSOR'

    .. code-block:: python
        :caption: Indexing and values

        from argopy import ArgoReferenceTable
        av = ArgoReferenceTable('SENSOR')

        # Values (or concept) within this reference table:
        ref.n_values    # Number of reference values
        ref.values      # List of reference value names
        ref.to_referencevalue() # Return a list of ArgoReferenceValue, for all values

        # Check for values:
        'CTD_TEMP_CNDC' in av  # Return True

        # Index by value key:
        ref['CTD_TEMP_CNDC']  # Return a ArgoReferenceValue instance

        # Index by value position in the table ordered list of values:
        ref[0]  # 1st ArgoReferenceValue instance
        ref[-1] # Last ArgoReferenceValue instance

    .. code-block:: python
        :caption: Methods

        from argopy import ArgoReferenceTable
        ref = ArgoReferenceTable('SENSOR')

        # Export table content to a pd.DataFrame:
        ref.to_dataframe()

        # Search methods (return a list of ArgoReferenceValue match):
        ref.search(name='RAMSES')         # Search in values name
        ref.search(definition='imaging')  # Search in values definition
        ref.search(long_name='TriOS')     # Search in values long name

    """
    __slots__ = ['nvs', 'identifier', 'parameter', 'long_name', 'description', 'version', 'date', 'uri', 'n_values', 'values', '_Vocabulary2Parameter', '_df', '_d']

    def __init__(self, identifier: str | None = None, *args, **kwargs) -> None:
        # Internal placeholders:
        self._Vocabulary2Parameter = Asset.load('vocabulary:mapping')['data']['Vocabulary2Parameter']
        self._df : pd.DataFrame | None = None  # Dataframe export
        self._d : dict[str, ArgoReferenceValue] | None = {}  # Dictionary of ArgoReferenceValue for all table concept

        if identifier in self._Vocabulary2Parameter:
            self.identifier = identifier
            self.parameter = self._Vocabulary2Parameter[identifier]
        elif identifier in self._Vocabulary2Parameter.values():
            self.parameter = identifier
            self.identifier = [k for k, v in self._Vocabulary2Parameter.items() if v == identifier][0]

        # Once we have an id in 'name' we can load raw data from NVS
        self.nvs = NVS().load_vocabulary(self.identifier)

        # And populate all attributes:
        Collection = [item for item in self.nvs['@graph'] if item['@type'] == 'skos:Collection'][0]

        self.long_name = Collection['skos:prefLabel']
        self.description = Collection["dc:description"]
        self.version = Collection['owl:versionInfo']
        self.date = pd.to_datetime(Collection['dc:date'])
        self.uri = Collection["@id"]
        self.n_values = len(Collection['skos:member'])

        # Two methods to retrieve the list of concept names:
        # From the skos:Collection list of members:
        # >>> values = [m['@id'].split("/")[-2] for m in Collection['skos:member']]
        # From skos:Concept in the @graph:
        # >>> values = [c['skos:altLabel'] for c in [item for item in self['@graph'] if item['@type'] == 'skos:Concept']]
        # We stick to Collection for consistency with other attributes gathering:
        self.values = [m['@id'].split("/")[-2] for m in Collection['skos:member']]
        self.values.sort()

    def __setattr__(self, attr, value):
        """Set attribute value, with read-only policy after instantiation for public attributes"""
        if attr in self.__slots__ and attr[0] != '_' and inspect.stack()[1][3] != '__init__':
            raise AttributeError(f"'{attr}' is read-only after instantiation.")
        ArgoReferenceTable.__dict__[attr].__set__(self, value)

    def __repr__(self):
        summary = [f"<argo.reference.table><{self.parameter}>"]
        summary.append(f'identifier: "{self.identifier}"')
        summary.append(f'long_name: "{self.long_name}"')
        summary.append(f"version: {self.version} ({self.date})")
        summary.append(f"uri: {self.uri}")
        summary.append(f'description: "{self.description}"')
        summary.append(f"values[{self.n_values}]: {ppliststr(self.values, n=10)}")
        return "\n".join(summary)

    def __contains__(self, item):
        return item in self.values

    def __len__(self):
        return self.n_values

    def __iter__(self):
        for ii, v in enumerate(self.values):
            yield self[ii]

    def __getitem__(self, key):
        """Get a reference value"""
        ref_value = None
        try:
            ref_value = self.values[key]
        except:
            if key in self.values:
                ref_value = key
        if ref_value is not None:
            if self._d.get(ref_value, None) is None:
                # The naive method is to call ArgoReferenceValue(name, reference):
                # self._d.update({ref_value: ArgoReferenceValue(ref_value, reference=self.identifier)})
                # But this is very slow for large reference tables because it requires one NVS fetch for
                # each concept.
                # This naive method is not necessary since all concepts data are already in self.nvs.
                data = [item for item in self.nvs['@graph'] if item['@type'] == 'skos:Concept' and item['skos:altLabel'] == ref_value]
                if len(data) == 1:
                    self._d.update({ref_value: ArgoReferenceValue.from_dict(data=data[0])})
            return self._d[ref_value]
        raise ValueError(f"Unknown index '{key}'")

    def to_referencevalue(self):
        return [self[ii] for ii, v in enumerate(self.values)]

    def to_dataframe(self, columns : list[str] | None = None) -> DataFrame | None:
        """Export all reference values to a :class:`pd.DataFrame`

        Default columns are given by :class:`ArgoReferenceValue` attributes.

        Parameters
        ----------
        columns: list[str] | None, optional, default=None
            Column names to insert into the output. By default, None, will include all available :class:`ArgoReferenceValue` attributes.

        Returns
        -------
        :class:`pd.DataFrame`
        """

        # Note that we could create a dataframe directly from self.nvs json data
        # But we want to stick to using ArgoReferenceValue attributes
        # so that there is only place determining how to map nvs json jargon onto user-friendly facade
        # i.e. the ArgoReferenceValue.
        if columns is None:
            cols = ArgoReferenceValue_attributes
        else:
            cols = []
            for c in columns:
                if c not in ArgoReferenceValue_attributes:
                    raise OptionValueError(
                        f"Invalid columns name '{c}'. Valid values are: {ppliststr(ArgoReferenceValue_attributes)}")
                cols.append(c)
            if len(cols) == 0:
                raise OptionValueError(
                    f"No valid column names in '{ppliststr(columns)}'. Valid values are: {ppliststr(ArgoReferenceValue_attributes, last='or')}")

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

    def search(self, **kwargs):
        """Search in table list of :class:`ArgoReferenceValue` attributes

        Parameters
        ----------
        tuple(str, str)
            Attributes to search.

            Using the specific argument `output='df'` to return a :class:`pd.DataFrame`.

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
            av = ArgoReferenceTable('SENSOR')

            # Search in specific attributes:
            av.search(name='RAMSES')
            av.search(definition='imaging')
            av.search(long_name='TriOS')

            # Search in more than one attribute:
            av.search(name='RAMSES', deprecated=False)

            # Control output format:
            av.search(name='CTD', output='df')

        """
        # Note that we could implement search on self.nvs json data
        # But we want to stick to using ArgoReferenceValue attributes
        # so that there is only place determining how to map nvs json jargon onto user-friendly facade
        # i.e. the ArgoReferenceValue.

        # Get output format:
        output = None
        if kwargs.get('output', None) is not None:
            output = kwargs.get('output')
            kwargs.pop('output')

        # Search key validation:
        keys = [key for key in kwargs if key in ArgoReferenceValue_attributes]

        # Search using the dataframe view of this table:
        df = self.to_dataframe()
        filters = []
        for key in keys:
            if df[key].dtype in ['str', 'object']:
                filters.append(df[key].str.contains(str(kwargs[key]), regex=True, case=False))
            elif df[key].dtype == 'datetime64[ns]':
                print("No search method implemented for datetime")
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
