from typing import Literal, Union
import xarray as xr

from ..utils.optical_modeling import Z_euphotic, Z_iPAR_threshold
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor("optic")
class OpticalModeling(ArgoAccessorExtension):
    """Optical modeling of the upper ocean

    This extension provides methods to compute standard variables from optical modeling of the upper ocean.

    See Also
    --------
    :class:`optic.Zeu`, :class:`optic.Zpd`, :class:`optic.Z_iPAR_threshold`

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

        dsp.argo.optic.Zpd()

        dsp.argo.optic.Z_iPAR_threshold(threshold=15.)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Zeu(
        self,
        axis: str = "PRES",
        par: str = "DOWNWELLING_PAR",
        method: Literal["percentage", "KdPAR"] = "percentage",
        max_surface: float = 5.0,
        layer_min: float = 10.0,
        layer_max: float = 50.0,
        inplace: bool = False,
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

        inplace: bool, optional, default: False
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            If the ``inplace`` argument is True, dataset is modified in-place with new variable ``Zeu``.

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
        :class:`Dataset.argo.optic`, :meth:`argopy.utils.Z_euphotic`
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
                "definition": "Depth given by 4.605/KdPAR (attenuation coefficient)",
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
            return self._obj
        else:
            return da

    def Zopt(self, *args, **kwargs):
        """Compute first optical depth from depth of the euphotic zone, points to Zpd method"""
        return self.Zpd(*args, **kwargs)

    def Zpd(
        self,
        axis: str = "PRES",
        par: str = "DOWNWELLING_PAR",
        *args,
        **kwargs
    ):
        """Compute first optical depth from depth of the euphotic zone

        Parameters
        ----------
        args, kwargs:
            All arguments are passed to :class:`Dataset.argo.optic.Zeu`

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            If the ``inplace`` argument is True, dataset is modified in-place with new variables Zpd and Zeu.

        Notes
        -----
        The "first optical depth", which is approximately the layer that is seen by a satellite, is given by [1]_:

        .. math::
            Z_{pd} = \\frac{Z_{eu}}{4.6}

        References
        ----------
        .. [1] Morel, A. (1988), Optical modeling of the upper ocean in relation to its biogenous matter content (case
            I waters), J. Geophys. Res., 93(C9), 10749–10768, doi:10.1029/JC093iC09p10749.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            dsp.argo.optic.Zpd()
            dsp.argo.optic.Zpd(method='KdPAR', layer_min=10., layer_maz=50.) # Modify how Zeu is computed

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`Dataset.argo.optic.Zeu`,
        """
        inplace = kwargs.get("inplace", False)
        if "Zeu" in self._obj:
            Zeu = self._obj["Zeu"]
        else:
            if inplace:
                Zeu = self.Zeu(*args, **kwargs)["Zeu"]
            else:
                Zeu = self.Zeu(*args, **kwargs)
        # da = self._argo.reduce_profile(Z_firstoptic, params=[axis, par], **kwargs)
        da = Zeu / 4.6  # This is too simple to call on the reduce function

        # Attributes
        da.name = "Zpd"
        da.attrs = {
            "long_name": "First optical depth",
            "units": Zeu.attrs.get("units", "?"),
            "definition": "Depth given by Zeu/4.6",
        }
        if not inplace:
            if Zeu.attrs.get("max_surface", False):
                da.attrs["Zeu_method"] = "percentage"
                da.attrs["Zeu_max_surface"] = Zeu.attrs["max_surface"]
            if Zeu.attrs.get("layer_min", False):
                da.attrs["Zeu_method"] = "KdPAR"
                da.attrs["Zeu_layer_min"] = Zeu.attrs["layer_min"]
                da.attrs["Zeu_layer_max"] = Zeu.attrs["layer_max"]

        if inplace:
            self._obj["Zpd"] = da
            return self._obj
        else:
            return da

    def Z_iPAR_threshold(
        self,
        axis: str = "PRES",
        par: str = "DOWNWELLING_PAR",
        threshold: float = 15.0,
        tolerance: float = 5.0,
        inplace: bool = False,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Depth where PAR reaches some threshold value (closest point)

        This is the closest level $z$ in the vertical axis for which PAR is about a threshold value $t$, with some tolerance $\\epsilon$:

        .. math::

            z | abs(PAR(z) - t) < \epsilon

        A default value of 15 is used because it is the theorical value below which the Fchla is no longer
        quenched (For correction of NPQ purposes).

        Parameters
        ----------
        axis: str, optional, default='PRES'
            Name of the pressure axis to use.
        par: str, optional, default='DOWNWELLING_PAR'
            Name of the PAR variable to use.
        threshold: float, optional, default: 15.
            Target value for ``par``.
        tolerance: float, optional, default: 5.
            PAR value tolerance with regard to the target threshold. If the closest PAR value to ``threshold`` is distant by more than ``tolerance``, consider result invalid and return NaN.
        inplace: bool, optional, default: False
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            If the ``inplace`` argument is True, dataset is modified in-place with new variable ``Z_iPAR``.

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`argopy.utils.Z_iPAR_threshold`
        """
        if axis not in self._obj:
            raise ValueError(f"Missing '{axis}' in this dataset")
        if par not in self._obj:
            raise ValueError(f"Missing '{par}' in this dataset")
        kw = {"threshold": threshold, "tolerance": tolerance}
        da = self._argo.reduce_profile(Z_iPAR_threshold, params=[axis, par], **kw)

        # Attributes
        da.name = "Z_iPAR"
        da.attrs = {
            "long_name": "Depth where PAR=%0.2f" % threshold,
            "units": self._obj[axis].attrs.get("units", "?"),
            "threshold": threshold,
            "tolerance": tolerance,
        }
        if inplace:
            self._obj["Z_iPAR"] = da
            return self._obj
        else:
            return da

    def DCM(self):
        """

        The depth of the [Chla] maximum is searched between the first vertical level and ``max_depth``,
        300m by default, assuming that no phytoplankton [Chla] can develop below 300 m.

        A [Chla] profile is definitively qualified as a DCM if the maximum [Chla] value of the un-smoothed profile is
        greater than twice the median of the [Chla] values at depths above ``surface_layer`` (15. by default).

        References
        ----------
        .. [1] Cornec 2021 (https://doi. org/10.1029/2020GB006759) Section 2.4

        Returns
        -------

        """