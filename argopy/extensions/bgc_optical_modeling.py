from ..utils.optical_modeling import Z_euphotic
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor('optic')
class OpticalModeling(ArgoAccessorExtension):
    """Optical modeling of the upper ocean

    Examples
    --------
    .. code-block:: python
        :caption: Example 1

        >>> from argopy import DataFetcher
        >>> ds = Datafetcher(ds='bgc', mode='expert', params='DOWNWELLING_PAR').float(6901864).data
        >>> dsp = ds.argo.point2profile()
        >>> dsp.argo.optic.Zeu()

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Zeu(self, par='DOWNWELLING_PAR', axis='PRES'):
        """Compute depth of the euphotic zone

        This is the depth at which the downwelling irradiance is reduced to 1% of its surface value.

        The surface value is taken as the maximum `PAR` above `max_surface`.

        The downwelling irradiance is from the photosynthetically available radiation: PAR.
        """
        da = self._argo.reduce_profile(Z_euphotic, par, axis)
        da.name = 'Zeu'
        da.attrs = {'long_name': 'Depth of the euphotic zone',
                    'units': self._obj[axis].attrs.get('units', '?'),
                    'definition': 'Depth at which the downwelling irradiance is reduced to 1% of its surface value.'}
        return da
