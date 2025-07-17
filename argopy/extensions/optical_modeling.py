from typing import Literal, Union
import xarray as xr

from ..utils import optical_modeling as om
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor("optic")
class OpticalModeling(ArgoAccessorExtension):
    """Optical modeling of the upper ocean

    This extension provides methods to compute standard variables from optical modeling of the upper ocean.

    See Also
    --------
    :class:`optic.Zeu`, :class:`optic.Zpd`, :class:`optic.Z_iPAR_threshold`, :class:`optic.DCM`

    Examples
    --------
    .. code-block:: python
        :caption: Example

        from argopy import DataFetcher
        dsp = DataFetcher(ds='bgc', mode='expert', params='DOWNWELLING_PAR').float(6901864).data.argo.point2profile()

        # or:
        # from argopy import ArgoFloat
        # dsp = ArgoFloat(6901864).open_dataset('Sprof')

        dsp.argo.optic.Zeu()
        dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
        dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_maz=50.)

        dsp.argo.optic.Zpd()

        dsp.argo.optic.Z_iPAR_threshold(threshold=15.)

        dsp.argo.optic.DCM()
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
        inplace: bool = True,
    ):
        """Depth of the euphotic zone from PAR

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

        inplace: bool, optional, default: True
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            Zeu as :class:`xarray.DataArray` or, if the ``inplace`` argument is True, dataset is modified in-place with new variable ``Zeu``.

        See Also
        --------
        :class:`Dataset.argo.optic`, :meth:`argopy.utils.optical_modeling.Z_euphotic`

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


        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            dsp.argo.optic.Zeu()
            dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
            dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_maz=50.)

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
        da = self._argo.reduce_profile(om.Z_euphotic, params=[axis, par], **kw)

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
            if self._argo:
                self._argo.add_history("Added Zeu prediction")
            return self._obj
        else:
            return da

    def Zopt(self, *args, **kwargs):
        """First optical depth from depth of the euphotic zone, points to Zpd method"""
        return self.Zpd(*args, **kwargs)

    def Zpd(self, *args, **kwargs):
        """First optical depth from depth of the euphotic zone

        Parameters
        ----------
        args, kwargs:
            Since Zpd is a simple scaling of the euphotic depth, all arguments are directly passed to :class:`Dataset.argo.optic.Zeu`.

        inplace: bool, optional, default: True
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            Zpd as a :class:`xarray.DataArray` or, if the ``inplace`` argument is True, dataset is modified in-place with new variables Zpd and Zeu.

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`Dataset.argo.optic.Zeu`, :meth:`argopy.utils.optical_modeling.Z_firstoptic`

        Notes
        -----
        The "first optical depth", which is approximately the layer that is seen by a satellite, is given by [1]_:

        .. math::
            Z_{pd} = \\frac{Z_{eu}}{4.6}

        References
        ----------
        .. [1] Morel, A. (1988), Optical modeling of the upper ocean in relation to its biogenous matter content (case I waters), J. Geophys. Res., 93(C9), 10749–10768, doi:10.1029/JC093iC09p10749.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            dsp.argo.optic.Zpd()
            dsp.argo.optic.Zpd(method='KdPAR', layer_min=10., layer_maz=50.) # Modify how Zeu is computed
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
            if self._argo:
                self._argo.add_history("Added Zeu prediction")
            return self._obj
        else:
            return da

    def Z_iPAR_threshold(
        self,
        axis: str = "PRES",
        par: str = "DOWNWELLING_PAR",
        threshold: float = 15.0,
        tolerance: float = 5.0,
        inplace: bool = True,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Depth where PAR reaches some threshold value (closest point)

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
        inplace: bool, optional, default: True
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            Z_iPAR as a :class:`xarray.DataArray` or, if the ``inplace`` argument is True, dataset is modified in-place with new variable ``Z_iPAR``.

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`argopy.utils.optical_modeling.Z_iPAR_threshold`

        Notes
        -----
        This is the closest level $z$ in the vertical axis for which PAR is about a threshold value $t$, with some tolerance $\epsilon$:

        .. math::

            z | abs(PAR(z) - t) < \epsilon

        A default value of 15 is used because it is the theoretical value below which the Fchla is no longer
        quenched (For correction of NPQ purposes).
        """
        if axis not in self._obj:
            raise ValueError(f"Missing '{axis}' in this dataset")
        if par not in self._obj:
            raise ValueError(f"Missing '{par}' in this dataset")
        kw = {"threshold": threshold, "tolerance": tolerance}
        da = self._argo.reduce_profile(om.Z_iPAR_threshold, params=[axis, par], **kw)

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
            if self._argo:
                self._argo.add_history("Added Z_iPAR prediction")
            return self._obj
        else:
            return da

    def DCM(
        self,
        axis: str = "PRES",
        chla: str = "CHLA",
        bbp: str = "BBP700",
        max_depth: float = 300.0,
        resolution_threshold: float = 3.0,
        median_filter_size: int = 5,
        surface_layer: float = 15.0,
        inplace: bool = True,
        axis_bbp: str = "PRES",
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Search and qualify Deep Chlorophyll Maxima

        This method return the main characteristics and drivers of Deep Chlorophyll Maxima (DCM). Different drivers are possible because DCMs result from photo-acclimation or biomass accumulation, depending on the availability of light and nitrate. See notes below.

        Parameters
        ----------
        axis: str, optional, default='PRES'
            Name of the pressure axis to use for CHLA and BBP. If BBP is not on the same vertical axis as CHLA, you can specify which variable to use with the optional parameter ``axis_bbp``.
        chla: str, optional, default='CHLA'
            Name of the Chl-a concentration variable to use.
        bbp: str, optional, default='BBP700'
            Name of the particulate backscattering coefficient variable to use.
        inplace: bool, optional, default: True
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            DCM driver as a :class:`xarray.DataArray` or, if the ``inplace`` argument is True, dataset is modified in-place with new variable ``DCM``.

        Other Parameters
        ----------------
        max_depth: float, optional, default: 300.
            Maximum depth allowed for a deep CHLA maximum to be found.
        resolution_threshold: float, optional, default: 3.
            CHLA vertical axis resolution threshold below which a smoother is applied.
        median_filter_size: int, optional, default: 5
            Size of the :func:`scipy.ndimage.median_filter` filter used with CHLA.
        surface_layer: float, optional, default: 15.
            Depth value defining the surface layer above which twice the median CHLA value may qualify a DCM as such.

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`argopy.utils.optical_modeling.DCM`

        Notes
        -----
        Following [1]_ Section 2.4 "Identification and Classification of Deep Maxima Profiles", a DCM is
        identified and then qualified along this procedure:

        - the depth of the Chla maximum is searched between the first vertical level and ``max_depth``, 300m by default, assuming that no phytoplankton Chla can develop below 300 m.
        - A Chla profile is definitively qualified as a DCM if the maximum Chla value of the un-smoothed profile is greater than twice the median of the Chla values at depths above ``surface_layer`` (15. by default).

        If the vertical resolution is less than 3 db or meters, the Chla profile is smoothed using a :func:`scipy.ndimage.median_filter` filter.

        If the vertical resolution is less than 3 db or meters, the BBP profile is smoothed using a :func:`scipy.ndimage.median_filter` and a :func:`scipy.ndimage.uniform_filter1d` filters.

        References
        ----------
        .. [1] Cornec, M., Claustre, H., Mignot, A., Guidi, L., Lacour, L., Poteau, A., et al. (2021). Deep
        chlorophyll maxima in the global ocean: Occurrences, drivers and characteristics. Global Biogeochemical
        Cycles, 35, e2020GB006759. https://doi.org/10.1029/2020GB006759
        """

        def f(*args, **kwargs):
            typ, dpt, amp = om.DCM(*args, **kwargs)
            return "%3s" % typ

        if axis not in self._obj:
            raise ValueError(f"Missing '{axis}' in this dataset")
        if chla not in self._obj:
            raise ValueError(f"Missing '{chla}' in this dataset")
        if bbp not in self._obj:
            raise ValueError(f"Missing '{bbp}' in this dataset")

        kw = {
            "max_depth": max_depth,
            "resolution_threshold": resolution_threshold,
            "median_filter_size": median_filter_size,
            "surface_layer": surface_layer,
        }
        da = self._argo.reduce_profile(f, params=[chla, axis, bbp, axis_bbp], **kw)

        # Attributes
        da.name = "DCM"
        da.attrs = {
            "long_name": "Deep Chlorophyll Maximum",
            "definition": {
                "NO ": f"No DCM found in {axis}<={max_depth}",
                "DBM": f"DCM is associated with a biomass maximum in {bbp}",
                "DAM": "DCM is due to photo-acclimation",
            },
            "max_depth": max_depth,
            "resolution_threshold": resolution_threshold,
            "median_filter_size": median_filter_size,
            "surface_layer": surface_layer,
        }
        if inplace:
            self._obj["DCM"] = da
            if self._argo:
                self._argo.add_history("Added DCM prediction")
            return self._obj
        else:
            return da

    def DCM_depth(
        self,
        axis: str = "PRES",
        chla: str = "CHLA",
        bbp: str = "BBP700",
        max_depth: float = 300.0,
        resolution_threshold: float = 3.0,
        median_filter_size: int = 5,
        surface_layer: float = 15.0,
        inplace: bool = True,
        axis_bbp: str = "PRES",
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Depth of the Deep Chlorophyll Maxima

        This method return the depth of the Deep Chlorophyll Maxima, if any.

        Parameters
        ----------
        axis: str, optional, default='PRES'
            Name of the pressure axis to use for CHLA and BBP. If BBP is not on the same vertical axis as CHLA, you can specify which variable to use with the optional parameter ``axis_bbp``.
        chla: str, optional, default='CHLA'
            Name of the Chl-a concentration variable to use.
        bbp: str, optional, default='BBP700'
            Name of the particulate backscattering coefficient variable to use.
        inplace: bool, optional, default: True
            Should we return the new variable (False) or the dataset with the new variable added to it (True).

        Returns
        -------
        :class:`xarray.DataArray` or :class:`xarray.Dataset`
            DCM driver as a :class:`xarray.DataArray` or, if the ``inplace`` argument is True, dataset is modified in-place with new variable ``DCM``.

        Other Parameters
        ----------------
        max_depth: float, optional, default: 300.
            Maximum depth allowed for a deep CHLA maximum to be found.
        resolution_threshold: float, optional, default: 3.
            CHLA vertical axis resolution threshold below which a smoother is applied.
        median_filter_size: int, optional, default: 5
            Size of the :func:`scipy.ndimage.median_filter` filter used with CHLA.
        surface_layer: float, optional, default: 15.
            Depth value defining the surface layer above which twice the median CHLA value may qualify a DCM as such.

        See Also
        --------
        :class:`Dataset.argo.optic`, :class:`argopy.utils.optical_modeling.DCM`

        Notes
        -----
        Following [1]_ Section 2.4 "Identification and Classification of Deep Maxima Profiles", a DCM is
        identified and then qualified along this procedure:

        - the depth of the Chla maximum is searched between the first vertical level and ``max_depth``, 300m by default, assuming that no phytoplankton Chla can develop below 300 m.
        - A Chla profile is definitively qualified as a DCM if the maximum Chla value of the un-smoothed profile is greater than twice the median of the Chla values at depths above ``surface_layer`` (15. by default).

        If the vertical resolution is less than 3 db or meters, the Chla profile is smoothed using a :func:`scipy.ndimage.median_filter` filter.

        If the vertical resolution is less than 3 db or meters, the BBP profile is smoothed using a :func:`scipy.ndimage.median_filter` and a :func:`scipy.ndimage.uniform_filter1d` filters.

        References
        ----------
        .. [1] Cornec, M., Claustre, H., Mignot, A., Guidi, L., Lacour, L., Poteau, A., et al. (2021). Deep
        chlorophyll maxima in the global ocean: Occurrences, drivers and characteristics. Global Biogeochemical
        Cycles, 35, e2020GB006759. https://doi.org/10.1029/2020GB006759
        """

        def f(*args, **kwargs):
            typ, dpt, amp = om.DCM(*args, **kwargs)
            return dpt

        if axis not in self._obj:
            raise ValueError(f"Missing '{axis}' in this dataset")
        if chla not in self._obj:
            raise ValueError(f"Missing '{chla}' in this dataset")
        if bbp not in self._obj:
            raise ValueError(f"Missing '{bbp}' in this dataset")

        kw = {
            "max_depth": max_depth,
            "resolution_threshold": resolution_threshold,
            "median_filter_size": median_filter_size,
            "surface_layer": surface_layer,
        }
        da = self._argo.reduce_profile(f, params=[chla, axis, bbp, axis_bbp], **kw)

        # Attributes
        da.name = "DCM_depth"
        da.attrs = {
            "long_name": "Deep Chlorophyll Maximum Depth",
            "max_depth": max_depth,
            "resolution_threshold": resolution_threshold,
            "median_filter_size": median_filter_size,
            "surface_layer": surface_layer,
        }
        if inplace:
            self._obj["DCM_depth"] = da
            if self._argo:
                self._argo.add_history("Added DCM depth prediction")
            return self._obj
        else:
            return da
