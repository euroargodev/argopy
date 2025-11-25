from abc import ABC, abstractmethod
from typing import Callable


class NVSProto(ABC):
    online : bool | None = None
    """Are we getting data from NVS server (True) or static assets (False)"""

    uid: str = None
    """Unique instance ID"""

    def __repr__(self):
        props = [
            key
            for key in self.__dir__()
            if key[0] != "_" and not isinstance(getattr(self, key), Callable)
        ]
        props = sorted(props)
        props_str = [f"{prop}={getattr(self, prop)}" for prop in props]
        return f"NVS({', '.join(props_str)})"

    @abstractmethod
    def load_vocabulary(self):
        raise NotImplementedError

    @abstractmethod
    def load_concept(self):
        raise NotImplementedError

    # @property
    # def vocabulary(self):
    #     return self._vocabulary
