from .vocabulary import ArgoReferenceTable
from .concept import ArgoReferenceValue
from .nvs import NVS


class ArgoReference:
    """Argo collection of Reference Tables

    This class relies on :class:`ArgoReferenceTable` and :class:`ArgoReferenceValue`.

    This is intended to replace the deprecated :class:`ArgoNVSReferenceTables`.


    This class provides methods to search the collection of Argo Reference Tables.

    Notes
    -----
    .. code-block:: python
        :caption: Transition from deprecated APIs to new ones

        # Deprecated:
        from argopy import ArgoNVSReferenceTable
        # New API:
        from argopy import ArgoReference, ArgoReferenceTable


        nvs = ArgoNVSReferenceTables()

        nvs.tbl(25)  # Deprecated, to be replaced by new API
        nvs.tbl(25).to_dataframe()  # Hint for backward compatibility during transition phase
        ArgoReferenceTable('R25').to_dataframe()  # New API

        nvs.tbl_name(25)  # Deprecated, to be replaced by new API
        nvs.tbl_name(25).name  # Hint for backward compatibility during transition phase
        ArgoReferenceTable('R25').name  # New API

        nvs.all_tbl  # Modified, now return a dict[str, ArgoReferenceTable], not a dict[str, pd.DataFrame]
        nvs.all_tbl['ARGO_WMO_INST_TYPE'].to_dataframe()  # Hint for backward compatibility during transition phase

        nvs.all_tbl_name  # Modified, now return a dict[str, ArgoReferenceTable], not a dict[str, pd.DataFrame]
        (nvs.all_tbl_name['R08'].name, nvs.all_tbl_name['R08'].description, nvs.all_tbl_name['R08'].uri)  # hint for backward compatibility

        nvs.search()  # Deprecated, replaced by new API
        nvs.search_tables()  # New API

        nvs.search(txt, where='title')  # Deprecated, replaced by new API with key 'parameter'
        nvs.search_tables(parameter=txt)  # New API

    .. code-block:: python

        from argopy import ArgoReference

        ref = ArgoReference()

    """
    __slots__ = []

    def __init__(self, *args, **kwargs) -> None:
        NVS()
        pass

    def from_any(self, vocab: str, **kwargs) -> ArgoReferenceTable | ArgoReferenceValue:
        """

        Parameters
        ----------
        vocab: str
            Any string. It can be: Argo parameter name (e.g. 'SENSOR'), Argo parameter value (e.g. '2BP'), Argo Reference Table identifier (e.g. 'R01'). We will try to guess the most appropriate response.

        Returns
        -------

        """
        raise NotImplementedError