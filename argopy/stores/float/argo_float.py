"""
If client is online (connected to the web) we work with the 'online' implementation
otherwise we fall back on an offline implementation.

The choice is really meaningfull when the client is using a local host. In this case
we don't know if client intends to be online or offline, so we check and implement.

"""

import logging

from ...utils import isconnected


log = logging.getLogger("argopy.stores.ArgoFloat")


if isconnected():
    from .implementations.argo_float_online import ArgoFloatOnline as FloatStore

    log.info("Using ONLINE Argo Float implementation")
else:
    from .implementations.argo_float_offline import ArgoFloatOffline as FloatStore

    log.info("Using OFFLINE Argo Float implementation")


class ArgoFloat(FloatStore):
    """
    Main docstring for ArgoFloat
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
