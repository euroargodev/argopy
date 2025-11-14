from abc import ABC, abstractmethod
from collections import UserList
import warnings
import logging
import copy


log = logging.getLogger("argopy.utils.accessories")


class RegistryItem(ABC):
    """Prototype for possible custom items in a Registry"""

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def isvalid(self, item):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Not implemented")


class Registry(UserList):
    """A list manager that can validate item type

    Examples
    --------
    You can commit new entry to the registry, one by one:

        >>> R = Registry(name='file')
        >>> R.commit('meds/4901105/profiles/D4901105_017.nc')
        >>> R.commit('aoml/1900046/profiles/D1900046_179.nc')

    Or with a list:

        >>> R = Registry(name='My floats', dtype='wmo')
        >>> R.commit([2901746, 4902252])

    And also at instantiation time (name and dtype are optional):

        >>> R = Registry([2901746, 4902252], name='My floats', dtype=float_wmo)

    Registry can be used like a list.

    It is iterable:

        >>> for wmo in R:
        >>>     print(wmo)

    It has a ``len`` property:

        >>> len(R)

    It can be checked for values:

        >>> 4902252 in R

    You can also remove items from the registry, again one by one or with a list:

        >>> R.remove('2901746')

    """

    def _complain(self, msg):
        if self._invalid == "raise":
            raise ValueError(msg)
        elif self._invalid == "warn":
            warnings.warn(msg)
        else:
            log.debug(msg)

    def _isinstance(self, item):
        is_valid = isinstance(item, self.dtype)
        if not is_valid:
            self._complain("%s is not a valid %s" % (str(item), self.dtype))
        return is_valid

    def _wmo(self, item):
        return item.isvalid

    def __init__(
        self, initlist=None, name: str = "unnamed", dtype=str, invalid="raise"
    ):
        """Create a registry, i.e. a controlled list

        Parameters
        ----------
        initlist: list, optional
            List of values to register
        name: str, default: 'unnamed'
            Name of the Registry
        dtype: :class:`str` or dtype, default: :class:`str`
            Data type of registry content. Can be any data type, including 'wmo' or :class:`float_wmo`
        invalid: str, default: 'raise'
            Define what do to when a new item is not valid. Can be 'raise' or 'ignore'
        """
        self.name = name
        self._invalid = invalid
        if str(dtype).lower() in ["float_wmo", "wmo"]:
            from argopy.utils.wmo import float_wmo
            self._validator = self._wmo
            self._dtype = "float_wmo"
            self.dtype = float_wmo
        elif hasattr(dtype, "isvalid"):
            self._validator = dtype.isvalid
            self._dtype = str(dtype).lower()
            self.dtype = dtype
        else:
            self._validator = self._isinstance
            self._dtype = str(dtype).lower()
            self.dtype = dtype
        # else:
        #     raise ValueError("Unrecognised Registry data type '%s'" % dtype)

        if initlist is not None:
            initlist = self._process_items(initlist)
        super().__init__(initlist)

    def __repr__(self):
        summary = ["<argopy.registry>%s" % str(self.dtype)]
        summary.append("Name: %s" % self.name)
        N = len(self.data)
        msg = "Nitems: %s" % N if N > 1 else "Nitem: %s" % N
        summary.append(msg)
        if N > 0:
            items = [str(item) for item in self.data]
            # msg = format_oneline("[%s]" % "; ".join(items), max_width=120)
            msg = "[%s]" % "; ".join(items)
            summary.append("Content: %s" % msg)
        return "\n".join(summary)

    def _process_items(self, items):
        if not isinstance(items, list):
            items = [items]
        if self._dtype == "float_wmo":
            items = [self.dtype(item, errors=self._invalid) for item in items]
        return items

    def commit(self, values):
        """R.commit(values) -- append values to the end of the registry if not already in"""
        items = self._process_items(values)
        for item in items:
            if item not in self.data and self._validator(item):
                super().append(item)
        return self

    def append(self, value):
        """R.append(value) -- append value to the end of the registry"""
        items = self._process_items(value)
        for item in items:
            if self._validator(item):
                super().append(item)
        return self

    def extend(self, other):
        """R.extend(iterable) -- extend registry by appending elements from the iterable"""
        self.append(other)
        return self

    def remove(self, values):
        """R.remove(valueS) -- remove first occurrence of values."""
        items = self._process_items(values)
        for item in items:
            if item in self.data:
                super().remove(item)
        return self

    def insert(self, index, value):
        """R.insert(index, value) -- insert value before index."""
        item = self._process_items(value)[0]
        if self._validator(item):
            super().insert(index, item)
        return self

    def __copy__(self):
        # Called with copy.copy(R)
        return Registry(copy.copy(self.data), dtype=self.dtype)

    def copy(self):
        """Return a shallow copy of the registry"""
        return self.__copy__()
