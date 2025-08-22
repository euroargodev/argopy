from abc import abstractmethod
from typing import NoReturn


class ArgoFloatExtension:
    """Prototype for ArgoFloat extensions

    All extensions should inherit from this class

    This prototype makes available:

    - the :class:`ArgoFloat` instance as ``self._obj``
    """

    __slots__ = "_obj"

    def __init__(self, obj):
        self._obj = obj


class ArgoFloatPlotProto(ArgoFloatExtension):
    """Extension providing plot methods"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ArgoFloat.plot.trajectory(...)"
        )

    @abstractmethod
    def trajectory(self):
        raise NotImplementedError

    @abstractmethod
    def map(self):
        raise NotImplementedError

    @abstractmethod
    def scatter(self):
        raise NotImplementedError
