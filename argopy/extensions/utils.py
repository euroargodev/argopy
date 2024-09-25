"""

Disclosure
----------
The following methods `AccessorRegistrationWarning`, `_CachedAccessor` and `_register_accessor` are
sourced from https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py

The class `register_argo_accessor` is an adaption of the `register_dataset_accessor` from xarray.

"""

import warnings
from ..xarray import xr, ArgoAccessor


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration.

    Disclosure
    ----------
    This class was copied from [xarray](https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py)
    under Apache License 2.0
    """


class _CachedAccessor:
    """Custom property-like object (descriptor) for caching accessors.

    Disclosure
    ----------
    This class was copied from [xarray](https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py)
    under Apache License 2.0
    """

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.argo.canyon
            return self._accessor

        # Use the same dict as @pandas.util.cache_readonly.
        # It must be explicitly declared in obj.__slots__.
        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}

        try:
            return cache[self._name]
        except KeyError:
            pass

        try:
            accessor_obj = self._accessor(obj)
        except AttributeError:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            raise RuntimeError(f"error initializing {self._name!r} accessor.")

        cache[self._name] = accessor_obj
        return accessor_obj


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type {cls!r} is "
                "overriding a preexisting attribute with the same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_argo_accessor(name):
    """A decorator to register an accessor as a custom property on :class:`xarray.Dataset.argo` objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Examples
    --------
    Somewhere in your code, you can register a class inheriting from :class:`argopy.extensions.ArgoAccessorExtension`
    with this decorator::

        @register_argo_accessor('floats')
        class WorkWithWMO(ArgoAccessorExtension):

             def __init__(self, *args, **kwargs):
                 super().__init__(*args, **kwargs)
                 self._uid = argopy.utils.to_list(np.unique(self._obj["PLATFORM_NUMBER"].values))

             @property
             def wmo(self):
                 return self._uid

             @property
             def N(self):
                  return len(self.wmo)

    It will be available to an Argo dataset, like this::

        ds.argo.floats.N
        ds.argo.floats.wmo

    See also
    --------
    :class:`argopy.extensions.ArgoAccessorExtension`

    """
    return _register_accessor(name, ArgoAccessor)


class ArgoAccessorExtension:
    """Prototype for Argo accessor extensions

    All extensions should inherit from this class

    This prototype makes available:

    - the parent :class:`xarray.Dataset` instance as ``self._obj``
    - the :class:`Dataset.argo` instance as ``self._argo``

    See also
    --------
    :class:`argopy.extensions.register_argo_accessor`
    """

    def __init__(self, obj):
        if isinstance(obj, xr.Dataset):
            self._obj = obj  # Xarray object
            self._argo = None
        else:
            self._obj = obj._obj  # Xarray object from ArgoAccessor
            self._argo = obj
