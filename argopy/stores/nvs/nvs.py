import logging
from argopy.utils.checkers import isconnected


log = logging.getLogger("argopy.related.vocabulary.nvs")

if isconnected():
    from .implementations.online.nvs import NVS as Implementation

    log.info("Using ONLINE NVS implementation")
else:
    from .implementations.offline.nvs import NVS as Implementation

    log.info("Using OFFLINE NVS implementation")


class NVS(Implementation):
    """NVS data access manager

    Used by other classes to handle NVS download for a table/vocabulary/collection, a value/concept or a mapping.

    This class will always try to work with online data directly from a NVS server.

    But if **Argopy** is loaded offline, this class will fall back on using static assets and still return NVS data.

    Examples
    --------
    .. code-block:: python

        from argopy.stores import NVS

        nvs = NVS()

        # Load vocabularies (tables):
        nvs.load_vocabulary('R27')
        nvs.load_vocabulary_collection('R27') # A subset of data to briefly describe a vocabulary

        # Load concepts (values):
        nvs.load_concept('AANDERAA_OPTODE_3835')
        nvs.load_concept('1', rtid='R05')  # Need to specify the vocabulary id for a concept seen in more than one

        # Load mappings:
        nvs.load_mapping('R24', 'R23')

    Notes
    -----
    This class has a singleton design, i.e. only one instance creation is done and will be return on all subsequent instantiations.

    This implies that to modify the creation options, they must be defined on the first instanciation in the session, otherwise they will be ignored.


    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
