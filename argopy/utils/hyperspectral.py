"""
Low-level functions for hyperspectral sensors (eg: RAMSES)
"""
import warnings
import numpy as np
import xarray as xr
from typing import Dict, Any

from argopy.errors import InvalidDatasetStructure
from argopy.utils.mappers import map_vars_to_dict


def read_coefs(ds: xr.Dataset) -> Dict[str, Any]:
    """Read pre-deployment calibration coefficients for down irradiance spectrum

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        Argo float auxiliary metadata file

    Returns
    -------
    coefs: Dict[str, Any]
        All possible coefficients, typically: ['c0s', 'c1s', 'c2s', 'c3s', 'c4s', 'CalAQ', 'Back1', 'Back2']

    Raises
    ------
    :class:`InvalidDatasetStructure`
        Raised if not working with an appropriate dataset
    """
    if ds.attrs["title"] != "Argo float auxiliary metadata file":
        raise InvalidDatasetStructure(
            "This function is only for one float auxiliary metadata file"
        )

    if not ("PARAMETER" in ds and "PREDEPLOYMENT_CALIB_COEFFICIENT" in ds):
        raise InvalidDatasetStructure(
            "Cannot read pre-deployment calibration coefficients from this dataset (missing variables 'PREDEPLOYMENT_CALIB_COEFFICIENT')"
        )

    try:
        all_coefs = map_vars_to_dict(
            ds, "PARAMETER", "PREDEPLOYMENT_CALIB_COEFFICIENT"
        )
    except:  # noqa: E722
        raise InvalidDatasetStructure(
            "Cannot map 'PREDEPLOYMENT_CALIB_COEFFICIENT' onto all 'PARAMETER' names"
        )

    if "DOWN_IRRADIANCE_SPECTRUM" in all_coefs:
        coefs = all_coefs["DOWN_IRRADIANCE_SPECTRUM"]
    else:
        raise InvalidDatasetStructure(
            "Parameter 'DOWN_IRRADIANCE_SPECTRUM' is not available in this auxiliary metadata dataset"
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
    bin_size: int,
    pixel_binned: bool = False,
) -> np.ndarray:
    """Compute one RAMSES hyperspectral wavelengths array

    Implement the polynomial computation of the wavelengths, as described in the user manual "Processing
    BGC-Argo radiometric data at the DAC Level".

    Parameters
    ----------
    c0s: float
        Pre-deployment calibration coefficients for down irradiance spectrum
    c1s: float
        Pre-deployment calibration coefficients for down irradiance spectrum
    c2s: float
        Pre-deployment calibration coefficients for down irradiance spectrum
    c3s: float
        Pre-deployment calibration coefficients for down irradiance spectrum
    c4s: float
        Pre-deployment calibration coefficients for down irradiance spectrum

    pixel_start: int
        Starting index of the wavelength region of interest within the 1..244 pixels transmitted by the sensor. Only used if argument
        ``pixel_binned`` is True.
    pixel_stop: int
        Ending index of the wavelength region of interest within the 2..245 pixels transmitted by the sensor. Only used if argument
        ``pixel_binned`` is True.
    bin_size: int
        Size if the wavelength region of interest within the 0..255 pixels transmitted by the sensor. Only used if argument
        ``pixel_binned`` is True.

    pixel_binned: bool, default = False
        Should we return wavelengths using pixel-binned estimate, or not. See details of the computation below.

    Returns
    -------
    :class:`numpy.ndarray`
        A wavelengths array

    Notes
    -----
    For each pixel i (from 0 to 255), the wavelength of the pixel is computed from the sensor meta
    data coefficients ``c0s``, ``c1s``, ``c2s``, ``c3s`` and ``c4s`` according to:

    .. math::

        \lambda(i) = c0s + c1s(i+1) + c2s(i+1)^2 + c3s(i+1)^3 + c4s(i+1)^4

    where $i = 1 .. 255$.

    If **pixel_binned = False**:

    We consider that all 255 pixels i from 1 to 255 are transmitted by the float, including the dark pixels between
    DarkPixelBegin ... DarkPixelEnd indices. The N_VALUES dimension of the hyperspectral data is thus 255.

    If **pixel_binned = True**:

    We consider that the size of the wavelengths array depends on how the 255 pixels (the first DeadPixel 0 is
    always excluded) were binned.

    1.
        The wavelength region of interest is chosen by using the ``pixel_start`` and ``pixel_stop`` arguments:
        Typically:

        - ``pixel_start`` = $i_{min}$ = 5, but can be either from 1 to 244
        - ``pixel_stop`` = $i_{max}$ = 144, but can be either from 2 to 245

    2.
        The spectral resolution is possibly reduced by grouping sets of adjacent pixels using the ``bin_size``
        argument. Typically:

        - ``bin_size`` = $N_{pixel}$ = 2, but can be either 1, 2 or 4

        The new pixel's value returned by the float is the average of the binned pixels rounded to the minimum integer
        (and not the closest). Binning is performed by the float before transmission. Individual pixel data are lost
        (unless ``bin_size`` = 1).


    As a result, the N_VALUES dimension of the hyperspectral data is given by the following formula:

    .. math::

        N_{VALUES} = \\frac{i_{max} - i_{min} + 1}{N_{pixel}}

    The corresponding wavelengths :math:`\lambda_b(i_b)` or each binned pixels $i_b$ are calculated by bin-averaging the
    sensor's full wavelength scale, :math:`\lambda(i)`, from the equation above, into the same grouped pixels $i_b$ as
    determined by the configuration parameters:

    .. math::

        \lambda_b(i_b) = \\frac{1}{N_{pixel}} \\sum_{i=i_{min}+(i_b-1)N_{pixel}}^{i_{min}+i_b*N_{pixel}-1} \lambda(i)

    where $i_{b} = 1 .. N_{VALUES}$.


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
    if not pixel_binned:
        return wavelengths

    # Else, consider a pixel bin configuration (eg: PROVOR float):
    Imin = pixel_start
    Imax = pixel_stop
    Nbin = bin_size

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
