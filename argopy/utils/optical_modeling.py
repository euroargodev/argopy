"""
Utility module for optical modelling diagnostics.

These functions are not really meant to be used directly, they consume raw 1D array of data.

You should rather use the :class:`xarray.Dataset.argo.optic` extension.

"""

import numpy as np
from typing import Tuple, Annotated, Literal, TypeVar

from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d

import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1D = Annotated[npt.NDArray[DType], Literal["N"]]


try:
    import gsw

    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False


def Z_euphotic(
    axis: np.ndarray,
    par: np.ndarray,
    method: Literal["percentage", "KdPAR"] = "percentage",
    max_surface: float = 5.0,
    layer_min: float = 10.0,
    layer_max: float = 50.0,
) -> float:
    """Depth of the euphotic zone from unlabeled arrays of pressure and PAR

    Two methods are available:

    - **percentage**: Depth for which PAR is 1% that of the surface value, defined as the maximum above ``max_surface``
    - **KdPAR**: -log(0.01) times the PAR attenuation coefficient over the layer between ``layer_min`` and ``layer_max``

    See :class:`xarray.Dataset.argo.optic.Zeu` for more details on the methodology.

    Parameters
    ----------
    axis: numpy.ndarray, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, typically from the ``PRES`` parameter of an Argo float.
    par: array_like, 1 dimensional
        Photosynthetically available radiation, typically from the ``DOWNWELLING_PAR`` parameter of an Argo float.
    method: Literal['percentage', 'KdPAR'] = 'percentage'
        Computation method to use.

    max_surface: float, optional, default: 5.
        Used only with the ``percentage`` method.
        Maximum value of the vertical axis above which the maximum PAR value is considered surface values.

    layer_min: float, optional, default: 10.

    layer_max: float, optional, default: 50.
        Used only with the ``KdPAR`` method.
        Minimum and maximum values of the vertical axis over which to compute the PAR attenuation coefficient.

    Returns
    -------
    float
        Estimate of the euphotic layer depth

    See Also
    --------
    :class:`xarray.Dataset.argo.optic.Zeu`
    """
    idx = ~np.logical_or(np.isnan(axis), np.isnan(par))
    axis = axis[idx]
    par = par[idx]

    if method == "percentage":
        try:
            Surface_levels = np.where(axis <= max_surface)[0]
        except Exception:
            Surface_levels = np.ndarray()
            pass

        result = np.nan
        if Surface_levels.shape[0] > 0:
            Surface_value = np.max(par[Surface_levels])
            if np.any(par > (Surface_value / 100)):
                index_1_percent = np.argmin(np.abs(par - (Surface_value / 100)))
                result = axis[index_1_percent]

    elif method == "KdPAR":
        result = np.nan
        layer_index = (axis >= layer_min) & (axis <= layer_max)
        if np.any(layer_index):
            layer_size = (
                axis[layer_index][0] - axis[layer_index][-1]
            )  # 0 index is below (higher value) the -1 index
            Kd_layer = (
                -1
                / layer_size
                * (np.log(par[layer_index][0]) - np.log(par[layer_index][-1]))
            )
            result = -np.log(0.01) / Kd_layer

    return np.array(result)


def Z_firstoptic(*args, **kwargs) -> float:
    """First optical depth from unlabeled arrays of pressure and PAR

    See :class:`xarray.Dataset.argo.optic.Zpd` for more details on the methodology.

    Parameters
    ----------
    args, kwargs:
        All arguments are passed to :meth:`Z_euphotic`.

    Returns
    -------
    float
        Estimate of the first optical depth

    See Also
    --------
    :class:`xarray.Dataset.argo.optic.Zpd`, :meth:`Z_euphotic`, :mod:`argopy.utils.optical_modeling`
    """
    Zeu = Z_euphotic(*args, **kwargs)
    return Zeu / 4.6


def Z_iPAR_threshold(
    axis: np.ndarray, par: np.ndarray, threshold: float = 15.0, tolerance: float = 5.0
) -> float:
    """Depth where unlabelled array of PAR reaches some threshold value (closest point)

    The closest level in the vertical axis for which PAR is about a ``threshold`` value, with some tolerance.

    See :class:`xarray.Dataset.argo.optic.Z_iPAR_threshold` for more details on the methodology.

    Parameters
    ----------
    axis: array_like, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, typically from the ``PRES``
        parameter of an Argo float.
    par: array_like, 1 dimensional
        Photosynthetically available radiation, typically from the ``DOWNWELLING_PAR`` parameter of an Argo float.
    threshold: float, optional, default: 15.
        Target value for ``par``. We use 15 as the default because it is the theoretical value below which
        the Fchla is no longer quenched (For correction of NPQ purposes).
    tolerance: float, optional, default: 5.
        PAR value tolerance with regard to the target threshold. If the closest PAR value to ``threshold`` is distant by more than ``tolerance``, consider result invalid and return NaN.

    Returns
    -------
    float

    See Also
    --------
    :class:`xarray.Dataset.argo.optic.Z_iPAR_threshold`
    """
    # index = np.argmin(np.abs(par - 15))
    # Z_iPAR = axis[index]
    #
    # if not par[index] > 10 or not par[index] < 20:
    #     return np.nan
    # else:
    #     return Z_iPAR
    iz = np.argmin(np.abs(par - threshold))
    par_z = par[iz]
    result = axis[iz]
    if np.abs(par_z - threshold) >= tolerance:
        return np.nan
    else:
        return np.array(result)


def DCM(
    CHLA: np.ndarray,
    CHLA_axis: np.ndarray,
    BBP: np.ndarray,
    BBP_axis: np.ndarray,
    max_depth: float = 300.0,
    resolution_threshold: float = 3.0,
    median_filter_size: int = 5,
    surface_layer: float = 15.0,
    median_filter_size_bbp: int = 7,
    uniform_filter1d_size_bbp: int = 5,
) -> Tuple[Literal["NO ", "DBM", "DAM"], float, float]:
    """Search and qualify Deep Chlorophyll Maxima from unlabeled arrays of pressure and CHLA/BBP

    See :class:`xarray.Dataset.argo.optic.DCM` for more details on the methodology.

    Parameters
    ----------
    CHLA: array_like, 1 dimensional
        Chlorophyl-a concentration profile data.
    CHLA_axis: array_like, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, for CHLA. Typically, from the ``PRES``
        parameter of an Argo float.
    BBP: array_like, 1 dimensional
        Particulate backscattering coefficient profile data.
    BBP_axis: array_like, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, for BBP. Typically, from the ``PRES``
        parameter of an Argo float.
    max_depth: float, optional, default: 300.
        Maximum depth allowed for a deep CHLA maximum to be found.
    resolution_threshold: float, optional, default: 3.
        CHLA vertical axis resolution threshold below which a smoother is applied.
    median_filter_size: int, optional, default: 5
        Size of the :func:`scipy.ndimage.median_filter` filter used with CHLA.
    surface_layer: float, optional, default: 15.
        Depth value defining the surface layer above which twice the median CHLA value may qualify a DCM as such.

    Other Parameters
    ----------------
    median_filter_size_bbp: int, optional, default: 7
        Size of the :func:`scipy.ndimage.median_filter` filter used with BBP.
    uniform_filter1d_size_bbp: int, optional, default: 7
        Size of the :func:`scipy.ndimage.uniform_filter1d` filter used with BBP.

    Returns
    -------
    str[3], Literal['NO ', 'DBM', 'DAM']
        The type of Deep Chlorophyll Maxima (DCM). Possible values are:

        - 'NO ': No DCM found above ``max_depth``
        - 'DBM' : Deep Biomass Maximum
        - 'DAM' : Deep photoAcclimation Maximum
    float
        Depth of the DCM, from the un-smoothed profile
    float
        Amplitude the DCM: CHLA value, from the un-smoothed profile.

    See Also
    --------
    :class:`xarray.Dataset.argo.optic.DCM`
    """
    idx = ~np.logical_or(np.isnan(CHLA_axis), np.isnan(CHLA))
    CHLA_axis = CHLA_axis[idx]
    CHLA = CHLA[idx]

    # Possibly smooth the profile:
    if np.diff(CHLA_axis[CHLA_axis <= max_depth]).mean().round() < resolution_threshold:
        # Rolling median window:
        CHLA_smooth = median_filter(CHLA, median_filter_size, mode="nearest")
    else:
        CHLA_smooth = np.copy(CHLA)

    # Identify the CHLA maximum in the appropriate layer:
    layer_CHLA = CHLA_axis <= max_depth
    if ~np.any(layer_CHLA):
        return "NO", np.nan, np.nan
    Max_CHLA_depth = CHLA_axis[layer_CHLA][np.argmax(CHLA_smooth[layer_CHLA])]
    Max_CHLA = CHLA[layer_CHLA][np.argmax(CHLA_smooth[layer_CHLA])]

    # Qualify CHLA maximum as a DCM:
    if np.any(CHLA_axis <= surface_layer) and Max_CHLA > 2 * np.nanmedian(
        CHLA[CHLA_axis <= surface_layer]
    ):
        DCM_type = "DCM"
    else:
        return "NO", np.nan, np.nan

    # Check for a potential cooccurrence of the DCM depth with any deep peak of BBP:
    if DCM_type == "DCM":
        idx = ~np.logical_or(np.isnan(BBP_axis), np.isnan(BBP))
        BBP_axis = BBP_axis[idx]
        BBP = BBP[idx]

        if (
            np.diff(BBP_axis[BBP_axis <= max_depth]).mean().round()
            < resolution_threshold
        ):
            # Rolling median window 5, Rolling mean 7
            BBP_smooth = median_filter(BBP, median_filter_size_bbp, mode="nearest")
            BBP_smooth = uniform_filter1d(
                BBP_smooth, uniform_filter1d_size_bbp, mode="nearest"
            )
        else:
            BBP_smooth = np.copy(BBP)

        # BBP maximum was searched for from the smoothed BBP profile in a layer of 20 meters around the DCM
        layer_bbp = (BBP_axis >= Max_CHLA_depth - 20) & (
            BBP_axis <= Max_CHLA_depth + 20
        )
        # once the bbp maximum and depth were identified on the smoothed profile,
        # closest bbp measurements on the unsmoothed profile were accordingly identified.
        Max_bbp = BBP[layer_bbp][np.argmax(BBP_smooth[layer_bbp])]

        # The profile was defined as a DBM if the BBP maximum was more than 1.3
        # times the BBP minimum within the top 15 meters.
        if Max_bbp > 1.3 * np.min(BBP[BBP_axis <= surface_layer]):
            DCM_type = "DBM"  # Deep Biomass Maximum
        else:
            DCM_type = "DAM"  # Deep photoAcclimation Maximum

    if DCM_type == "NO":
        return "%3s" % DCM_type, np.nan, np.nan
    else:
        return "%3s" % DCM_type, Max_CHLA_depth, Max_CHLA


def MLD_Func(PRES, PSAL, TEMP, LAT, LON):
    """
    Parameters
    ----------
    Process potential density using gsw package

    Return MLD with Boyer Montégut method with threshold of σ(10m) + 0.03 kg.m-3

    """
    SA = gsw.SA_from_SP(PSAL, PRES, LON, LAT)
    CT = gsw.CT_from_t(SA, TEMP, PRES)
    density = gsw.sigma0(SA, CT)

    depth_density = PRES[~np.isnan(density)]
    density = density[~np.isnan(density)]

    if not any(depth_density < 10) or all(depth_density < 10):
        return np.nan

    else:
        index_10m = np.argmin(np.abs(depth_density - 10))
        density_at_10m = density[index_10m]

        index = np.min(
            np.where(density[index_10m:] > density_at_10m + 0.03) + index_10m
        )
        MLD = depth_density[index]

        return MLD


def time_UTC_tolocal(time_64, longitude):
    delta = 60 * longitude / 15  # 60 min = 15°

    local = time_64 + np.timedelta64(int(delta), "m")

    return local


def time_UTC_to_offset(time_64, longitude):
    if abs(longitude) <= 7.5:
        # return time_64
        return 0

    elif abs(longitude) > 7.5 and abs(longitude) < 15:
        # local = time_64 + np.timedelta64(int(1 * np.sign(longitude)), "h")
        # return local
        return 1 * np.sign(longitude)

    else:
        offset_hours = int((abs(longitude) + 7.5) / 15)
        # local = time_64 + np.timedelta64(int(offset_hours * np.sign(longitude)), "h")
        # return local
        return offset_hours * np.sign(longitude)


def get_solar_angle(LATITUDE, LONGITUDE, JULD):
    import pvlib

    location = pvlib.location.Location(LATITUDE, LONGITUDE)

    solar_position = location.get_solarposition(JULD)

    return solar_position["apparent_elevation"].values
