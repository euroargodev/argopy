from abc import abstractmethod
from typing import NoReturn
import logging

from ...utils import register_accessor
from ...errors import InvalidDatasetStructure, OptionValueError
from ...utils import to_list
from .implementations.index_s3 import search_s3


log = logging.getLogger("argopy.stores.index.extensions")


def register_ArgoIndex_accessor(name, store):
    """A decorator to register an accessor as a custom property on :class:`ArgoIndex` objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.
    store: :class:`ArgoIndex`

    Examples
    --------
    .. code-block:: python

        @register_ArgoIndex_accessor('query')
        class SearchEngine(ArgoIndexExtension):

             def __init__(self, *args, **kwargs):
                 super().__init__(*args, **kwargs)

             def wmo(self, WMOs):
                 return WMOs

    It will be available to an ArgoIndex object, like this::

        ArgoIndex().query.wmo(WMOs)
    """
    return register_accessor(name, store)


class ArgoIndexExtension:
    """Prototype for ArgoIndex extensions

    All extensions should inherit from this class

    This prototype makes available:

    - the :class:`ArgoIndex` instance as ``self._obj``
    """

    __slots__ = "_obj"

    def __init__(self, obj):
        self._obj = obj


class ArgoIndexSearchEngine(ArgoIndexExtension):
    """Extension providing search methods to query index entries

    All search methods can be combined with the :meth:`ArgoIndex.query.compose` method, see examples.

    Examples
    --------
    .. code-block:: python
        :caption: List of search methods

        from argopy import ArgoIndex
        idx = ArgoIndex()

        idx.query.wmo
        idx.query.cyc
        idx.query.wmo_cyc

        idx.query.lon
        idx.query.lat
        idx.query.date
        idx.query.lat_lon
        idx.query.box

        idx.query.params
        idx.query.parameter_data_mode

        idx.query.profiler_type
        idx.query.profiler_label

        idx.query.institution_code
        idx.query.institution_name
        idx.query.dac

    .. code-block:: python
        :caption: Composition of queries

        from argopy import ArgoIndex
        idx = ArgoIndex(index_file='bgc-s')

        idx.query.compose({'box': BOX, 'wmo': WMOs})
        idx.query.compose({'box': BOX, 'params': 'DOXY'})
        idx.query.compose({'box': BOX, 'params': 'DOXY'})
        idx.query.compose({'box': BOX, 'params': (['DOXY', 'DOXY2'], {'logical': 'and'})})
        idx.query.compose({'params': 'DOXY', 'profiler_label': 'ARVOR'})

    Note that composing query with:

    - ``wmo`` and ``cyc`` is slower than using the ``wmo_cyc`` method
    - ``lon`` and ``lat`` is slower than using the ``lon_lat`` method
    - ``lon``, ``lat`` and ``date`` is slower than using the ``box`` method

    """

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoIndex.query cannot be called directly. Use "
            "an explicit search method, e.g. ArgoIndex.query.box(...)"
        )

    @abstractmethod
    def wmo(self):
        """Search index for floats defined by WMO

        Parameters
        ----------
        WMOs: list(int) or list(str)
            List of WMOs to search

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.wmo(2901746)
            idx.query.wmo([2901746, 4902252])
        """
        raise NotImplementedError

    @abstractmethod
    def cyc(self):
        """Search index for cycle numbers

        Parameters
        ----------
        CYCs: list(int) or list(str)
            List of cycle number to search

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.cyc(1)
            idx.query.cyc([1,2])
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def wmo_cyc(self):
        """Search index for floats defined by their WMO and specific cycle numbers

        Parameters
        ----------
        WMOs: list(int) or list(str)
            List of WMOs to search

        CYCs: list(int) or list(str)
            List of cycle number to search

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.search_wmo_cyc(2901746, 12)
            idx.search_wmo_cyc([2901746, 4902252], [1,2])
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def lon(self):
        """Search index for a meridional band

        Parameters
        ----------
        BOX : list()
            An index box to search Argo records for.

        Returns
        -------
        :class:`ArgoIndex`

        Warnings
        --------
        Only longitude bounds are used from the index box.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def lat(self):
        """Search index for a zonal band

        Parameters
        ----------
        BOX : list()
            An index box to search Argo records for.

        Returns
        -------
        :class:`ArgoIndex`

        Warnings
        --------
        Only latitude bounds are used from the index box.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def date(self):
        """Search index for a date range

        Parameters
        ----------
        BOX : list()
            An index box to search Argo records for.

        Returns
        -------
        :class:`ArgoIndex`

        Warnings
        --------
        Only date bounds are used from the index box.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.date([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def lon_lat(self):
        """Search index for a rectangular longitude/latitude domain

        Parameters
        ----------
        BOX : list()
            An index box to search Argo records for.

        Returns
        -------
        :class:`ArgoIndex`

        Warnings
        --------
        Only lat/lon bounds are used from the index box.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.lon_lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def box(self):
        """Search index for a box: a rectangular latitude/longitude domain and time range

        Parameters
        ----------
        BOX : list()
            An index box to search Argo records for.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='core')

            idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def params(self):
        """Search index for one or a list of parameters

        Parameters
        ----------
        PARAMs: str or list(str)
            A string or a list of strings to search Argo records for in the PARAMETERS columns of BGC profiles index.
        logical: str, default='and'
            Indicate to search for all (``and``) or any (``or``) of the parameters.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='bgc-s')

            idx.query.params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])
            idx.query.params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'], logical='or')

        Warnings
        --------
        This method is only available for index following the ``bgc-s``, ``bgc-b`` and ``aux`` conventions.

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def parameter_data_mode(self):
        """Search index for profiles with a parameter in a specific data mode

        Parameters
        ----------
        PARAMs: dict
            A dictionary with parameters as keys, and data mode as a string or a list of strings
        logical: str, default='and'
            Indicate to search for all (``and``) or any (``or``) of the parameters data mode. This operator applies
            between each parameter.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='bgc-s')

            idx.query.parameter_data_mode({'TEMP': 'D'})
            idx.query.parameter_data_mode({'BBP700': 'D'})
            idx.query.parameter_data_mode({'DOXY': ['R', 'A']})
            idx.query.parameter_data_mode({'BBP700': 'D', 'DOXY': 'D'}, logical='or')

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def profiler_type(self):
        """Search index for profiler types

        The list of valid types is given by IDs of `Argo reference table 8 <http://vocab.nerc.ac.uk/collection/R08/current/>`_.

        Parameters
        ----------
        profiler_type: str, list(str)
            List of profiler types to search for.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex()

            idx.query.profiler_type(845)

        .. code-block:: python
            :caption: List valid types

            from argopy import ArgoNVSReferenceTables
            valid_types = ArgoNVSReferenceTables().tbl(8)['altLabel']

        See Also
        --------
        :class:`ArgoIndex.query.profiler_label`
        """
        raise NotImplementedError("Not implemented")

    def profiler_label(self, profiler_label: str, nrows=None, composed=False):
        """Search index for profiler types with a given string in their long name

        Will search for string occurrences in the preferred label of `Argo reference table 8 <http://vocab.nerc.ac.uk/collection/R08/current/>`_.

        Parameters
        ----------
        profiler_label: str, list(str)
            The string (not exact) to be found in profiler preferred labels.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex()

            idx.query.profiler_label('ARVOR')

        .. code-block:: python
            :caption: List valid labels

            from argopy import ArgoNVSReferenceTables
            valid_labels = ArgoNVSReferenceTables().tbl(8)['prefLabel']

        See Also
        --------
        :class:`ArgoIndex.query.profiler_type`
        """
        def checker(profiler_label):
            if "profiler_type" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for profiler labels in this index)")
            log.debug("Argo index searching for profiler label '%s' ..." % profiler_label)
            profiler_label = to_list(profiler_label)
            profiler_type = []
            for ptype, long_name in self._obj._r8.items():
                for label in profiler_label:
                    if label in long_name:
                        profiler_type.append(ptype)
            return profiler_label, profiler_type

        def namer(profiler_label):
            self._obj.search_type.pop('PTYPE')
            return {"PLABEL": profiler_label}

        def composer(profiler_type):
            return self.profiler_type(profiler_type, nrows=nrows, composed=True)

        profiler_label, profiler_type = checker(profiler_label)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(profiler_type)
        if not composed:
            self._obj.search_type = namer(profiler_label)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(profiler_label))
            return search_filter

    @abstractmethod
    def institution_code(self, institution_code,  nrows=None, composed=False):
        """Search index for institution codes

        The list of valid codes is given by IDs of `Argo reference table 4 <http://vocab.nerc.ac.uk/collection/R04/current/>`_.

        Parameters
        ----------
        institution_code: str, list(str)
            List of institution codes to search for.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex()

            idx.query.institution_code('IF')
            idx.query.institution_code(['IF', 'JA'])

        .. code-block:: python
            :caption: List valid codes

            from argopy import ArgoNVSReferenceTables
            valid_codes = ArgoNVSReferenceTables().tbl(4)['altLabel']

        See Also
        --------
        :class:`ArgoIndex.query.institution_name`, :class:`ArgoIndex.query.dac`
        """
        raise NotImplementedError("Not implemented")

    @search_s3
    def institution_name(self, institution_name: str, nrows=None, composed=False):
        """Search index for institutions with a given string in their long name

        Will search for string occurrences in the preferred label of `Argo reference table 4 <http://vocab.nerc.ac.uk/collection/R04/current/>`_.

        Parameters
        ----------
        institution_name: str, list(str)
            The string (not exact) to be found in institution preferred labels.

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex()

            idx.query.institution_name('Canada')
            idx.query.institution_name(['Canada', 'germany'])

        .. code-block:: python
            :caption: List valid names

            from argopy import ArgoNVSReferenceTables
            valid_names = ArgoNVSReferenceTables().tbl(4)['prefLabel']

        See Also
        --------
        :class:`ArgoIndex.query.institution_code`, :class:`ArgoIndex.query.dac`
        """
        def checker(institution_name):
            if "institution" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for institution name in this index)")
            log.debug("Argo index searching for institution name '%s' ..." % institution_name)
            institution_name = to_list(institution_name)
            institution_code = []
            for code, long_name in self._obj._r4.items():
                for label in institution_name:
                    if label.lower() in long_name.lower():
                        institution_code.append(code)
            if len(institution_code) == 0:
                valid_names = ", ".join(self._obj.valid.institution_name)
                raise OptionValueError(f"No valid institution name found in {institution_name}. Valid names are any string in: '{valid_names}'")
            else:
                return institution_name, institution_code

        def namer(institution_name):
            self._obj.search_type.pop('INST_CODE')
            return {"INST_NAME": institution_name}

        def composer(institution_code):
            return self.institution_code(institution_code, nrows=nrows, composed=True)

        institution_name, institution_code = checker(institution_name)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(institution_code)
        if not composed:
            self._obj.search_type = namer(institution_name)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(institution_name))
            return search_filter

    @abstractmethod
    def dac(self, dac,  nrows=None, composed=False):
        """Search index for DAC

        Parameters
        ----------
        dac: str, list(str)
            One or more DAC names to look for (based on file paths).

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex()

            idx.query.dac('coriolis')
            idx.query.dac(['aoml', 'meds'])

        See Also
        --------
        :class:`ArgoIndex.query.institution_code`, :class:`ArgoIndex.query.institution_name`
        """
        raise NotImplementedError("Not implemented")

    def compose(self, query: dict, nrows=None):
        """Compose query with multiple search methods

        Parameters
        ----------
        query: dict
            A dictionary with search method as keys and search criteria as values

        Returns
        -------
        :class:`ArgoIndex`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex
            idx = ArgoIndex(index_file='bgc-s')

            idx.query.compose({'box': BOX, 'wmo': WMOs})
            idx.query.compose({'box': BOX, 'params': 'DOXY'})
            idx.query.compose({'box': BOX, 'params': 'DOXY'})
            idx.query.compose({'box': BOX, 'params': (['DOXY', 'DOXY2'], {'logical': 'and'})})
            idx.query.compose({'params': 'DOXY', 'profiler_label': 'ARVOR'})

        """
        self._obj.search_type = {}
        filters = []
        for entry, arg in query.items():
            searcher = getattr(self, entry)
            if not isinstance(arg, tuple):
                filter = searcher(arg, composed=True)
            else:
                kw = arg[1]
                kw.update({'composed': True})
                filter = searcher(arg[0], **kw)
            filters.append(filter)
        self._obj.search_filter = self._obj._reduce_a_filter_list(filters, op='and')
        self._obj.run(nrows=nrows)
        return self._obj


class ArgoIndexSearchValidProto(ArgoIndexExtension):
    """Extension providing valid values for search queries and validation methods"""


class ArgoIndexPlotProto(ArgoIndexExtension):
    """Extension providing plot methods"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoIndex.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ArgoIndex.plot.bar(...)"
        )

    def get_title(self, index=False):
        title = "Argo index '%s'" % self._obj.index_file
        if hasattr(self._obj, "search") and not index:
            title += ": %s" % self._obj.search_type
        return title

    @abstractmethod
    def trajectory(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def bar(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError("Not implemented")
