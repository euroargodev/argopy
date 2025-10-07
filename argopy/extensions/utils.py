from ..xarray import xr, ArgoAccessor
from ..utils import register_accessor


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
    return register_accessor(name, ArgoAccessor)


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
