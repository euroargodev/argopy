import warnings
from .xarray import ArgoAccessor


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _CachedAccessor:
    """Custom property-like object (descriptor) for caching accessors."""

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


def register_argodataset_accessor(name):
    """Register a custom property on xarray.Dataset.argo objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Examples
    --------
    In your library code:

    >>> @xr.register_argodataset_accessor("canyon_med")
    ... class CanyonMED:
    ...     def __init__(self, xarray_obj):
    ...         self._obj = xarray_obj
    ...
    ...     def predict(self):
    ...         # Make NN predictions
    ...         pass
    ...

    Back in an interactive IPython session:

    >>> ds.argo.canyon_med.predict()


    See Also
    --------
    register_dataarray_accessor
    """
    return _register_accessor(name, ArgoAccessor)
