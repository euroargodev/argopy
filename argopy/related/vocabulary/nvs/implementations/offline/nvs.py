from typing import Any

from ......utils import Asset
from ...spec import NVSProto
from ...utils import concept2vocabulary


class NVS(NVSProto):
    online = False

    _instance: "NVS | None" = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "NVS":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        if not self._initialized:
            self._initialized = True
        self.uid = id(self)
        self.nvs = "<local.static.assets>"

    def vocabulary2url(self, rtid: str):
        return f"vocabulary:offline:{rtid}"

    def load_vocabulary(self, rtid: str):
        url = self.vocabulary2url(rtid)
        return Asset().load(url)['data']

    def concept2url(self, conceptid: str, rtid: str | None = None):
        if rtid is None:
            reftable = concept2vocabulary(conceptid)
            if reftable is None:
                raise ValueError('Invalid Concept')
            if len(reftable) > 1:
                raise ValueError(
                    f"This Concept appears in more than one Vocabulary: {reftable}. You must specified with the 'rtid' argument which one to use.")
            else:
                rtid = reftable[0]
        return f"vocabulary:offline:{rtid}:{conceptid}"

    def load_concept(self, conceptid: str, rtid: str | None = None):
        url = self.concept2url(conceptid, rtid)
        return Asset().load(url)['data']