import numpy as np
from typing import Literal

from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d

from typing import Annotated, Literal, TypeVar
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
    """Compute depth of the euphotic zone from unlabeled arrays of pressure and PAR

    Two methods are available:

    - **percentage**: Depth for which PAR is 1% that of the surface value, defined as the maximum above ``max_surface``
    - **KdPAR**: -log(0.01) times the PAR attenuation coefficient over the layer between ``layer_min`` and ``layer_max``

    Parameters
    ----------
    axis: numpy.ndarray, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, typically from the ``PRES`` parameter of an Argo float.
    par: np.ndarray, 1 dimensional
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
    float:
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
        except:
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


def Z_firstoptic(*args, **kwargs):
    """Compute first optical depth from depth of the euphotic zone

    Parameters
    ----------
    args, kwargs:
        All arguments are passed to :class:`Dataset.argo.optic.Zeu`.

    Returns
    -------
    :class:`xarray.DataArray` or :class:`xarray.Dataset`
        If the ``inplace`` argument is True, dataset is modified in-place with new variables Zpd and Zeu.

    See Also
    --------
    :class:`xarray.Dataset.argo.optic.Zpd`
    """
    Zeu = Z_euphotic(*args, **kwargs)
    return Zeu / 4.6


def Z_iPAR_threshold(
        axis: np.ndarray,
        par: np.ndarray,
        threshold: float = 15.0,
        tolerance: float = 5.0
):
    """Depth where PAR value = threshold (closest point)

    The closest level in the vertical axis for which PAR is about a ``threshold`` value, with some tolerance.

    Parameters
    ----------
    axis: np.ndarray, 1 dimensional
        Vertical axis values, pressure or depth, positive, increasing downward, typically from the ``PRES``
        parameter of an Argo float.
    par: np.ndarray, 1 dimensional
        Photosynthetically available radiation, typically from the ``DOWNWELLING_PAR`` parameter of an Argo float.
    threshold: float, optional, default: 15.
        Target value for ``par``. We use 15 as the default because it is the theorical value below which
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


def MLD_Func(PRES, PSAL, TEMP, LAT, LON):
    """
    Parameters
    ----------
    Process potential density using gsw package

    return MLD wth Boyer Montégut method wth threshold of σ(10m) + 0.03 kg.m-3

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


def get_solar_angle(LATITUDE, LONGITUDE, JULD):
    import pvlib

    location = pvlib.location.Location(LATITUDE, LONGITUDE)

    solar_position = location.get_solarposition(JULD)

    return solar_position["apparent_elevation"].values


def time_UTC_to_offset(time_64, longitude):
    if abs(longitude) <= 7.5:
        # return time_64
        return 0

    elif abs(longitude) > 7.5 and abs(longitude) < 15:
        local = time_64 + np.timedelta64(int(1 * np.sign(longitude)), "h")
        # return local
        return 1 * np.sign(longitude)

    else:
        offset_hours = int((abs(longitude) + 7.5) / 15)
        local = time_64 + np.timedelta64(int(offset_hours * np.sign(longitude)), "h")
        # return local
        return offset_hours * np.sign(longitude)


def get_DCM(CHLA, depth_CHLA, BBP, depth_BBP):
    """
    Cornec 2021 (https://doi. org/10.1029/2020GB006759) Section 2.4
    Parameters
    ----------
    CHLA, BBP, associated depth arrays :

    Returns
    -------
    DCM type (Deep Chlorophyll Maxima)
        'NO'
        'DBM' : Associated with a biomass maximum
        'DAM' : Due to photoacclimation
    """

    # Rolling median window 5
    CHLA_smooth = median_filter(CHLA, 5, mode="nearest")

    # The depth of the [Chla] maximum was then searched for between 0 and 300 m,
    # assuming that no phyto- plankton [Chla] can develop below 300 m
    layer_CHLA = depth_CHLA <= 300
    Max_CHLA_depth = depth_CHLA[layer_CHLA][np.argmax(CHLA_smooth[layer_CHLA])]
    # the closest [Chla] measurements on the unsmoothed profile were accordingly identified.
    Max_CHLA = CHLA[layer_CHLA][np.argmax(CHLA_smooth[layer_CHLA])]

    # The profile was definitively qualified as a DCM if the maximum [Chla] value of
    # the unsmoothed profile was greater than twice the median of the [Chla] values in the 15 first meters
    if Max_CHLA > 2 * np.median(CHLA[depth_CHLA <= 15]):
        DCM_type = "DCM"
    else:
        # Otherwise, it was qualified as NO.
        DCM_type = "NO"

    # potential cooccurrence of the DCM depth with any deep peak of b bp
    if DCM_type == "DCM":
        # Rolling median window 5, Rolling mean 7
        BBP_smooth = median_filter(BBP, 7, mode="nearest")
        BBP_smooth = uniform_filter1d(BBP_smooth, 5, mode="nearest")

        # bbp maximum was searched for from the smoothed b bp profile in a layer of 20 meters around the DCM
        layer_bbp = (depth_BBP >= Max_CHLA_depth - 20) & (
            depth_BBP <= Max_CHLA_depth + 20
        )
        # once the bbp maximum and depth were identified on the smoothed profile,
        # closest bbp measurements on the unsmoothed profile were accordingly identified.
        Max_bbp = BBP[layer_bbp][np.argmax(BBP_smooth[layer_bbp])]

        # The profile was de- fined as a DBM if the bbp maximum was more than 1.3
        # times the bbp minimum within the top 15 meters.
        if Max_bbp > 1.3 * np.min(BBP[depth_BBP <= 15]):
            DCM_type = "DBM"  # Deep Biomass Maximum
        else:
            DCM_type = "DAM"  # Deep photoAcclimation Maximum

    return DCM_type
