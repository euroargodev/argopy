import numpy as np
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from argopy.stores.float.extensions import ArgoFloatRamsesProto

# from argopy.errors import InvalidDatasetStructure
from argopy.utils import hyperspectral
from argopy.stores.float.utils import ramses as ramses_utils


class ArgoFloatRamses(ArgoFloatRamsesProto):
    """Extension providing access to the RAMSES censor parameters/variables

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoFloat

        af = ArgoFloat(1902578, aux=True)  # mission-specific pixels
        # af = ArgoFloat(3902759, aux=True)  # launch config pixels

        af.ramses.coefs
        af.ramses.binned_pixel
        af.ramses.cycle_strides

        af.ramses.launch_wavelengths
        af.ramses.wavelengths
        af.ramses.wavelengths_by_cycle(3)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Can we proceed ?
        ramses_utils.has_meta_aux(self._obj)

        # Load pre-deployment calibration coefficients:
        self._coefs = hyperspectral.read_coefs(self._obj.dataset("meta_aux"))

        # Which formulae do we need to use ?
        self._binned_pixel = ramses_utils.floatype_formulae(self._obj)

        # Init private placeholders
        self._cycle_strides = None

    @property
    def cycle_strides(self):
        """Pixel indexes for all cycles"""
        if self._cycle_strides is None:
            self._cycle_strides = ramses_utils.read_pixel_strides(self._obj)
        return self._cycle_strides

    @property
    def launch_wavelengths(self) -> np.ndarray:
        """Wavelengths based on launch configuration parameter values

        Returns
        -------
        np.ndarray
            Array with wavelengths in nm
        """

        # Load "launch" pixel indexes (not mission dependents):
        default_strides = ramses_utils.read_default_pixel_strides(self._obj)

        return hyperspectral.compute_ramses_wavelengths(
            self.coefs["c0s"],
            self.coefs["c1s"],
            self.coefs["c2s"],
            self.coefs["c3s"],
            self.coefs["c4s"],
            default_strides["PixelBegin"],
            default_strides["PixelEnd"],
            default_strides["BinningSize"],
            self.binned_pixel,
        )

    def wavelengths_by_cycle(self, cycle_number: int) -> np.ndarray:
        """Wavelengths for a specific cycle

        Returns
        -------
        np.ndarray
            Array with wavelengths in nm
        """

        # Load pixel indexes for this cycle:
        strides = ramses_utils.read_pixel_strides_by_cycle(self._obj, cycle_number)

        return hyperspectral.compute_ramses_wavelengths(
            self.coefs["c0s"],
            self.coefs["c1s"],
            self.coefs["c2s"],
            self.coefs["c3s"],
            self.coefs["c4s"],
            strides["PixelBegin"],
            strides["PixelEnd"],
            strides["BinningSize"],
            self.binned_pixel,
        )

    @property
    def wavelengths(self) -> Dict[int, np.ndarray]:
        """Wavelengths for all cycles

        Returns
        -------
        Dict[int, np.ndarray]
            A dictionary of wavelengths in nm as values and cycle numbers as keys
        """

        # Create a dict of arguments to parallelize computation:
        items = {}
        for cyc in self._obj.CYCLE_NUMBERS:
            items[cyc] = (
                self.coefs["c0s"],
                self.coefs["c1s"],
                self.coefs["c2s"],
                self.coefs["c3s"],
                self.coefs["c4s"],
                self.cycle_strides[cyc]["PixelBegin"],
                self.cycle_strides[cyc]["PixelEnd"],
                self.cycle_strides[cyc]["BinningSize"],
                self.binned_pixel,
            )

        def fct(cyc, item):
            return {cyc: hyperspectral.compute_ramses_wavelengths(*item)}

        # Parallel (multi-threading) computation of wavelengths:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fct, cyc, items[cyc]) for cyc, item in items.items()
            ]
            for future in futures:
                r = future.result()
                results.update(r)
            return results
