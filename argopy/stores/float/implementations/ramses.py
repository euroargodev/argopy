import numpy as np
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from argopy.stores.float.extensions import ArgoFloatRamsesProto

from argopy.utils import hyperspectral
from argopy.stores.float.utils import ramses as ramses_utils


class ArgoFloatRamses(ArgoFloatRamsesProto):
    """Extension providing access to the RAMSES sensor parameters/variables

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoFloat

        af = ArgoFloat(1902578, aux=True)

        # Attributes provide information about the computation:
        af.ramses.coefs          # Dict[str, float | List[float]]
        af.ramses.binned_pixel   # Bool
        af.ramses.cycle_strides  # Dict[int, Dict[str, int]]

        # Methods to compute wavelengths in 3 complementary ways:
        af.ramses.launch_wavelengths      # np.ndarray
        af.ramses.wavelengths_by_cycle(3) # np.ndarray
        af.ramses.wavelengths             # Dict[int, np.ndarray]

    See Also
    --------
    :mod:`argopy.utils.hyperspectral`

    Notes
    -----
    This extension remains an experimental feature that can change its signature without deprecation warnings.
    In particular, methods may return different data types in the future.

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
    def cycle_strides(self)->Dict[int, Dict[str, int]]:
        """Pixel indexes for all cycles

        See Also
        --------
        :class:`ArgoFloat.ramses_utils.read_pixel_strides`
        """
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
        launch_strides = ramses_utils.read_launch_pixel_strides(self._obj)

        return hyperspectral.compute_ramses_wavelengths(
            self.coefs["c0s"],
            self.coefs["c1s"],
            self.coefs["c2s"],
            self.coefs["c3s"],
            self.coefs["c4s"],
            launch_strides["PixelBegin"],
            launch_strides["PixelEnd"],
            launch_strides["BinningSize"],
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
