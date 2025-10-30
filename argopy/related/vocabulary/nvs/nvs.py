import logging
from ....utils import isconnected


log = logging.getLogger("argopy.related.vocabulary.nvs")

if isconnected():
    from .implementations.online.nvs import NVS as Implementation

    log.info("Using ONLINE NVS implementation")
else:
    from .implementations.offline.nvs import NVS as Implementation

    log.info("Using OFFLINE NVS implementation")


class NVS(Implementation):
    """Load json data from the NVS

    Used by other classes to handle NVS json download for a table/vocabulary or a value/concept.

    This class will always try to work with online data directly from the server.

    But if **argopy** is loaded offline, this class will fall back on using static assets and still return NVS data.

    .. code-block:: python

        nvs = NVS()
        nvs.load_vocabulary('R27')
        nvs.load_concept('AANDERAA_OPTODE_3835')
        nvs.load_concept('1', rtid='R05')  # Need to specify the vocabulary for a concept seen in more than one

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
