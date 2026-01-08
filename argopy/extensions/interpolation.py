import numpy as np
import xarray as xr
from typing import Literal
from dataclasses import dataclass

from argopy.extensions.utils import register_argo_accessor, ArgoAccessorExtension
from argopy.utils.computers import pchip_interpolation_remap, linear_interpolation_remap


@dataclass(frozen=True)
class StandardLevels:
    """A dataclass for standard pressure levels"""
    value: np.array
    tolerance: np.array = None

    def __post_init__(self):
        if np.any(sorted(self.value) != self.value) | (np.any(self.value < 0.0)):
            raise ValueError(
                "Standard levels must be a list or a numpy array of positive and sorted values"
            )

    @classmethod
    def from_product(cls, product_name: str):
        valid_products = ["easyoneargolite"]
        if product_name.lower() not in valid_products:
            raise ValueError(
                f"Invalid pre-defined standard levels '{product_name}'. Must be one in: {valid_products}"
            )

        if product_name.lower() == "easyoneargolite":
            pTolerance = None
            std = [2.0]
            [std.append(z) for z in np.arange(5, 250, 5.0)]
            [std.append(z) for z in np.arange(250, 350, 10.0)]
            [std.append(z) for z in np.arange(350, 500, 25.0)]
            [std.append(z) for z in np.arange(500, 1000, 50.0)]
            [std.append(z) for z in np.arange(1000, 2000, 100.0)]
            [std.append(z) for z in np.arange(2000, 6000, 200.0)]

            # [std.append(z) for z in np.arange(5, 100, 5.)]
            # [std.append(z) for z in np.arange(100, 350, 10.)]
            # [std.append(z) for z in np.arange(350, 500, 25.)]
            # [std.append(z) for z in np.arange(500, 2000, 50.)]
            # [std.append(z) for z in np.arange(2000, 6000, 200.)]

            std.append(6000.0)
            std_lev = np.array(std, dtype=np.float32)

            scale = 3.0
            pTolerance = [scale * (std_lev[0] - 0.0)]  # 1st level near surface
            for ip, _ in enumerate(std_lev[0:-1]):
                local_spacing = std_lev[ip + 1] - std_lev[ip]
                pTolerance.append(scale * local_spacing)
            pTolerance = np.array(pTolerance, dtype=np.float32)

            return cls(value=std_lev, tolerance=pTolerance)


def read_da_list(dsp: xr.Dataset) -> tuple[xr.DataArray]:
    """Get some lists of xr.DataArray"""

    # List all variables to interpolate:
    datavars = [
        dv
        for dv in list(dsp.variables)
        if set(["N_LEVELS", "N_PROF"]) == set(dsp[dv].dims)
           and "QC" not in dv
           and "ERROR" not in dv
           and "DATA_MODE" not in dv
    ]
    # List coordinates to preserve:
    coords = [dv for dv in list(dsp.coords)]

    # List variables depending on N_PROF only
    # (hence not needing interpolation and that will be replicated in the output dataset):
    solovars = [
        dv
        for dv in list(dsp.variables)
        if dv not in datavars
           and dv not in coords
           and "QC" not in dv
           and "ERROR" not in dv
    ]

    return datavars, coords, solovars


@register_argo_accessor("interp")
class Interpolator(ArgoAccessorExtension):
    """Interpolation methods

    Examples
    --------
    .. code-block:: python
        :caption: Example

        from argopy import DataFetcher


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spl(
        self,
        levels: list[list, np.array, str] = "EasyOneArgoLite",
        axis: str = "PRES",
        method: Literal["mrst-pchip", "pchip", "linear"] = "pchip",
    ):
        """Interpolate measurements to standard pressure levels

        Parameters
        ----------
        levels: list or np.array or str, optional, default = 'EasyOneArgoLite'
            Standard pressure levels used for interpolation. It has to be 1-dimensional and monotonic.

            Some pre-defined levels are available with keywords:

            - ``EasyOneArgoLite`` (default)

            See Notes for details.

        axis: str, default: ``PRES``
            The dataset variable to use as pressure axis. This can be ``PRES`` or ``PRES_ADJUSTED``.

        method: str, LiteralString['pchip', 'mrst-pchip', 'linear'], default='pchip'
            The interpolation method:

            - ``pchip``: Apply the Piecewise Cubic Hermite Interpolating Polynomial method on all parameters
            - ``linear``: Apply :class:`scipy.interpolate.interp1d` on all parameters
            - ``mrst-pchip``: Apply:

                - Multiply-Rotated Salinity-Temperature PCHIP Method on temperature and salinity (Barker and McDougall, 2020)
                - Piecewise Cubic Hermite Interpolating Polynomial method on all other parameters

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        Pre-defined standard pressure levels are available for:

        - ``EasyOneArgoLite``: 107 vertical levels from 2dbar to 6000 decibar.
            The spacings between the vertical levels are:

            - 2dbar@0-2dbar,
            - 5dbar@5-250dbar,
            - 10dbar@250-350dbar,
            - 25dbar@350-500dbar,
            - 50dbar@500-1000dbar,
            - 100dbar@1000-2000dbar,
            - 200dbar@2000-6000dbar.

        - ``GLORYS``
        - ``ISAS``
        - ``IPRC``
        - ``ECCO V4``
        - ``RG``
        - ``MOAA``
        - ``CORA5``

        References
        ----------
        Barker, P. M., and T. J. McDougall, 2020: Two Interpolation Methods Using Multiply-Rotated
        Piecewise Cubic Hermite Interpolating Polynomials. J. Atmos. Oceanic Technol., 37, 605-619,
        https://doi.org/10.1175/JTECH-D-19-0211.1.

        """
        if axis not in ["PRES", "PRES_ADJUSTED"]:
            raise ValueError("'axis' option must be 'PRES' or 'PRES_ADJUSTED'")

        if isinstance(levels, str):
            std_lev = StandardLevels.from_product(levels)
        elif (type(levels) is np.ndarray) | (type(levels) is list):
            std_lev = StandardLevels(np.array(levels))
        else:
            raise ValueError(
                "Standard levels must be a string for pre-defined values or list or a numpy array of positive and sorted values."
            )

        # Handle input data structure: points or profiles
        # (we need to work with profiles)
        to_point = False
        if self._argo._type == "point":
            to_point = True
            this_dsp = self._argo.point2profile()
        else:
            this_dsp = self._obj.copy(deep=True)

        # Add new vertical dimensions, this has to be in the datasets to apply ufunc later
        this_dsp["Z_LEVELS"] = xr.DataArray(std_lev.value, dims={"Z_LEVELS": std_lev.value})

        # Get some lists of xr.DataArray:
        datavars, coords, solovars = read_da_list(this_dsp)

        # Create a dataset that will hold interpolation results:
        ds_out = xr.Dataset()

        if method == "linear":
            # Profile parameters are linearly interpolated, one after the other

            for da in datavars:
                ds_out[da] = linear_interpolation_remap(
                    this_dsp[axis],
                    this_dsp[da],
                    this_dsp["Z_LEVELS"],
                    z_dim="N_LEVELS",
                    z_regridded_dim="Z_LEVELS",
                    output_dim=f"{axis}_INTERPOLATED",
                )
                ds_out[da].attrs = this_dsp[da].attrs  # Preserve attributes
                if "long_name" in ds_out[da].attrs:
                    ds_out[da].attrs["long_name"] = f"Interpolated {ds_out[da].attrs['long_name']}"

        elif method == "pchip":
            # Profile parameters are interpolated with pchip, one after the other

            for da in datavars:
                ds_out[da] = pchip_interpolation_remap(
                    this_dsp[axis],
                    this_dsp[da],
                    this_dsp["Z_LEVELS"],
                    z_dim="N_LEVELS",
                    z_regridded_dim="Z_LEVELS",
                    output_dim=f"{axis}_INTERPOLATED",
                    zTolerance=std_lev.tolerance,
                )
                ds_out[da].attrs = this_dsp[da].attrs  # Preserve attributes
                if "long_name" in ds_out[da].attrs:
                    ds_out[da].attrs["long_name"] = f"Interpolated {ds_out[da].attrs['long_name']}"


        elif method == "mrst-pchip":
            raise NotImplementedError

        ds_out[f"{axis}_INTERPOLATED"].attrs = this_dsp[axis].attrs
        if "long_name" in ds_out[f"{axis}_INTERPOLATED"].attrs:
            ds_out[f"{axis}_INTERPOLATED"].attrs["long_name"] = f"Standard {axis} levels"

        for sv in solovars:
            ds_out[sv] = this_dsp[sv]

        for co in coords:
            ds_out.coords[co] = this_dsp[co]

        ds_out = ds_out.drop_vars(["N_LEVELS", "Z_LEVELS"])
        ds_out = ds_out[np.sort(ds_out.data_vars)]
        ds_out = ds_out.argo.cast_types()
        ds_out.attrs = self._argo.attrs.copy()  # Preserve original attributes
        ds_out.argo.add_history(f"Interpolated on standard {axis} levels with the '{method}' method (see Argopy documentation for details)")

        return ds_out
