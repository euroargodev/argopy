import warnings
import numpy as np
import xarray as xr
from typing import Dict, Any

from argopy.errors import InvalidDatasetStructure
from argopy.utils.mappers import map_vars_to_dict


def read_coefs(ds: xr.Dataset) -> Dict[str, Any]:
    """Read predeployment calibration coefficients for down irradiance spectrum

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        Argo float auxiliary metadata file

    Returns
    -------
    coefs: Dict[str, Any]

    Raises
    ------
    :class:`InvalidDatasetStructure`
        Raised if not working with an appropriate dataset
    """
    if ds.attrs["title"] != "Argo float auxiliary metadata file":
        raise InvalidDatasetStructure(
            "This function is only for one float auxiliary metadata file"
        )

    if "PARAMETER" in ds and "PREDEPLOYMENT_CALIB_COEFFICIENT" in ds:
        sub_ds = ds[
            ["PARAMETER", "PREDEPLOYMENT_CALIB_COEFFICIENT"]
        ]  # both have dim: (N_PARAMS, )
    else:
        raise InvalidDatasetStructure(
            "Cannot read pre-deployment calibration coefficients from this dataset (missing variables 'PREDEPLOYMENT_CALIB_COEFFICIENT')"
        )

    try:
        all_coefs = map_vars_to_dict(
            sub_ds, "PARAMETER", "PREDEPLOYMENT_CALIB_COEFFICIENT"
        )
    except:  # noqa: E722
        raise InvalidDatasetStructure(
            "Cannot map 'PREDEPLOYMENT_CALIB_COEFFICIENT' onto all 'PARAMETER' names"
        )

    if "DOWN_IRRADIANCE_SPECTRUM" in all_coefs:
        coefs = all_coefs["DOWN_IRRADIANCE_SPECTRUM"]
    else:
        raise InvalidDatasetStructure(
            "'DOWN_IRRADIANCE_SPECTRUM' is not a parameter in this dataset"
        )

    coefs_d: Dict[str, Any] = {}
    for line in coefs.split(";"):
        # n_coefs = line.count("=")
        if "[" not in line:
            cc = {
                k.strip(): float(v)
                for k, v in (pair.split("=") for pair in line.split(","))
            }
            coefs_d.update(cc)
        else:
            key, value = line.strip().split("=", 1)
            cc = {key: [float(x) for x in value.strip("[]").split(",")]}
            coefs_d.update(cc)
    assert len(coefs_d.keys()) == coefs.count("=")

    return coefs_d


def compute_ramses_wavelengths(
    c0s: float,
    c1s: float,
    c2s: float,
    c3s: float,
    c4s: float,
    pixel_start: int,
    pixel_stop: int,
    pixel_bin: int,
    binned_pixel: bool = False,
) -> np.ndarray:
    """Compute RAMSES hyperspectral wavelengths

    polynomial computation of the wavelengths

    Parameters
    ----------
    c0s: float, c1s: float, c2s: float, c3s: float, c4s: float
        Predeployment calibration coefficients for down irradiance spectrum
    pixel_start: int, pixel_stop: int, pixel_bin: int

    binned_pixel: bool, default = False
        Should we return wavelengths using pixel-binned estimate, or not.

    Notes
    -----
    For each pixel i (from 0 to 255), the wavelength of the pixel is computed from the sensor meta
    data c0s, c1s, c2s, c3s and c4s according to:

    WAVELENGTH_DOWN_IRR(i) = c0s+c1s(i+1)+c2s(i+1)2+c3s(i+1)3+c4s(i+1)4

    If with_pixel = False: all 255 pixels i from 1 to 255 are transmitted by the float, including
    the dark pixels between DarkPixelBegin ... DarkPixelEnd indices. The N_VALUES dimension of the
    hyperspectral data is 255.

    """
    wavelengths = np.zeros((255,))
    for i in range(1, 255 + 1):
        wavelengths[i - 1] = (
            c0s
            + c1s * (i + 1)
            + c2s * (i + 1) ** 2
            + c3s * (i + 1) ** 3
            + c4s * (i + 1) ** 4
        )

    # Possibly stop here (eg: APEX float):
    if not binned_pixel:
        return wavelengths

    # else, consider the pixel configuration (eg: PROVOR float):

    # Imin=5 "5 is when the first pixel is pixel 0 "the dead pixel" / discussion with Edouard we stay as it is "
    Imin = pixel_start

    # Imax=144 "144 is when the first pixel is pixel 0 "the dead pixel" / discussion with Edouard we stay as it is "
    Imax = pixel_stop

    Nbin = pixel_bin
    N_VALUES = (Imax - Imin + 1) / Nbin

    if N_VALUES != np.fix(N_VALUES):
        warnings.warn(
            f"RAMSES wavelengths binning (Imax-Imin+1)/Nbin = {N_VALUES} is not an integer; the last partial bin is dropped."
        )
        N_VALUES = np.floor(N_VALUES)
    N_VALUES = int(N_VALUES)

    # Bin-averaging the sensor’s full wavelength scale
    waveb = np.zeros((N_VALUES,))
    for ib in range(0, N_VALUES):
        istr = Imin + ib * Nbin
        istp = Imin + (ib + 1) * Nbin - 1
        # wave = 0.0
        # for i in range(istr-1, istp):
        #     wave += wavelengths[i]
        # waveb[ib] = wave/Nbin
        waveb[ib] = np.mean(wavelengths[istr - 1 : istp])

    return waveb
