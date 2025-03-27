from typing import Literal

from ..utils.optical_modeling import Z_euphotic
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor("optic")
class OpticalModeling(ArgoAccessorExtension):
    """Optical modeling of the upper ocean

    See Also
    --------
    :class:`optic.Zeu`

    Examples
    --------
    .. code-block:: python
        :caption: Example

        from argopy import DataFetcher
        ds = Datafetcher(ds='bgc', mode='expert', params='DOWNWELLING_PAR').float(6901864).data
        dsp = ds.argo.point2profile()

        dsp.argo.optic.Zeu()
        dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
        dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_maz=50.)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Zeu(
        self,
        axis="PRES",
        par="DOWNWELLING_PAR",
        method: Literal["percentage", "KdPAR"] = "percentage",
        max_surface: float = 5.0,
        layer_min: float = 10.0,
        layer_max: float = 50.0,
        inplace=False,
    ):
        """Compute depth of the euphotic zone from PAR

        PAR is the photosynthetically available radiation

        Two methods are available (see details below):

        - **percentage**: Depth for which PAR is 1% that of the surface value, defined as the maximum above ``max_surface``
        - **KdPAR**: -ln(0.01) times the inversed PAR attenuation coefficient over the layer between ``layer_min`` and ``layer_max``

        Parameters
        ----------
        axis: str, optional, default='PRES'
            Name of the pressure axis to use.
        par: str, optional, default='DOWNWELLING_PAR'
            Name of the PAR variable to use.
        method: str, ['percentage', 'KdPAR'] = 'percentage'
            Computation method to use.

        max_surface: float, optional, default: 5.
            Used only with the ``percentage`` method.
            Maximum value of the vertical axis above which the maximum PAR value is considered surface values.

        layer_min: float, optional, default: 10.

        layer_max: float, optional, default: 50.
            Used only with the ``KdPAR`` method.
            Minimum and maximum values of the vertical axis over which to compute the PAR attenuation coefficient.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            dsp.argo.optic.Zeu()
            dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
            dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_maz=50.)

        Notes
        -----
        The euphotic depth is estimated using the first method **percentage** directly from the vertical PAR profile
        as the depth for which PAR is 1% that of the surface value:

        .. math::
            I_0 = max(I(z\\leq\\text{max_surface}))

            Z_e = z | I(z) = 0.01\\,I_0

        But the euphotic depth can also be estimated using the exponential decay of light with depth, described by Beer's Law [1]_:

        .. math::
            I(z) = I_0 \exp(-K_{PAR}\\,z)

        If we solve for $I(Z_e)=0.01 I_0$ we get:

        .. math::
            Z_e = -\\frac{\\ln(0.01)}{K_{PAR}} = \\frac{4.605}{K_{PAR}}

        where the attenuation coefficient $K_{PAR}$ is for a homogeneous layer $z_1<z_2$ the most appropriately given by [2]_:

        .. math::
            K_{PAR} = -\\frac{ln(PAR(z_2))-ln(PAR(z_1))}{z_2-z_1}

        Using the method **KdPAR**, the homogeneous layer is set by the ``layer_min`` and ``layer_max`` arguments.

        References
        ----------
        .. [1] Kirk, J.T., 1994. Light and photosynthesis in aquatic ecosystems. Cambridge university press.
        .. [2] Kpar: An optical property associated with ambiguous values. J. Lake Sci., 21 (2009), pp. 159-164

        See Also
        --------
        :class:`optic`, :meth:`argopy.utils.Z_euphotic`
        """
        if axis not in self._obj:
            raise ValueError(f"Missing '{axis}' in this dataset")
        if par not in self._obj:
            raise ValueError(f"Missing '{par}' in this dataset")

        # Computation:
        if method == "percentage":
            kw = {"method": "percentage", "max_surface": max_surface}
            attrs = {
                "definition": "Depth at which the downwelling irradiance is reduced to 1% of its surface value.",
                "max_surface": max_surface,
            }
        elif method == "KdPAR":
            kw = {"method": "KdPAR", "layer_min": layer_min, "layer_max": layer_max}
            attrs = {
                "definition": "Depth at which PAR attenuation coefficient is reduced by -log(0.01)",
                "layer_min": layer_min,
                "layer_max": layer_max,
            }
        da = self._argo.reduce_profile(Z_euphotic, params=[axis, par], **kw)

        # Attributes
        da.name = "Zeu"
        da.attrs = {
            **{
                "long_name": "Depth of the euphotic zone",
                "units": self._obj[axis].attrs.get("units", "?"),
            },
            **attrs,
        }
        if inplace:
            self._obj["Zeu"] = da
        else:
            return da
