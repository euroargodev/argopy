"""

RAMSES high-level utilities relying on a :class:`ArgoFloat` instance.

(Hence these functions cannot be located to the low-level argopy.utils.hyperspectral module).

"""

import warnings
from typing import Dict, Literal
from argopy.errors import InvalidDatasetStructure


DEFAULT_STRIDES = {
    "PixelBegin": 1,
    "PixelEnd": 255,
    "BinningSize": 1,
}
"""The pixel strides to use by default"""


def has_meta_aux(af: "ArgoFloat") -> None:
    if "meta_aux" not in af.ls_datasets():
        raise InvalidDatasetStructure(
            "Cannot compute wavelengths without auxiliary metadata. You should at least use the aux=True option to create an ArgoFloat instance including auxiliary data."
        )


def floatype_formulae(af: "ArgoFloat") -> bool:
    """Which formulae do we need to use ?"""
    return "PROVOR" in af.metadata["platform"]["type"]


def read_launch_pixel_strides(af: "ArgoFloat") -> Dict[str, int]:
    def read(pixel: Literal["PixelBegin", "PixelEnd", "BinningSize"]) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"

        if cfg_param in af.auxlaunchconfig:
            pixel_value = af.auxlaunchconfig[cfg_param]
        elif cfg_param in af.launchconfig:
            warnings.warn(
                f"Unexpectedly found {cfg_param} in the launch config parameters (not auxiliary)"
            )
            pixel_value = af.launchconfig[cfg_param]
        else:
            plist = [p for p in af.auxlaunchconfig.parameters if "Ramses" in p]
            raise InvalidDatasetStructure(
                f"'{cfg_param}' cannot be found in the meta-aux (nore meta) configuration parameters. RAMSES related parameters are [{', '.join(plist)}]"
            )

        return pixel_value

    # Load "launch" pixel indexes (not mission dependents):
    strides = {
        "PixelBegin": read("PixelBegin"),
        "PixelEnd": read("PixelEnd"),
        "BinningSize": read("BinningSize"),
    }

    return strides


def read_pixel_strides_by_cycle(af: "ArgoFloat", cycle_number) -> Dict[str, int]:
    def read(pixel: Literal["PixelBegin", "PixelEnd", "BinningSize"], cycle) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"

        if cfg_param in af.auxconfig:
            pixel_value = af.auxconfig.for_cycles(cfg_param, cycle)
        elif cfg_param in af.config:
            warnings.warn(
                f"Unexpectedly found {cfg_param} in the config parameters (not auxiliary)"
            )
            pixel_value = af.config.for_cycles(cfg_param, cycle)
        else:
            return None

        return pixel_value

    # Load default/launch pixel indexes:
    try:
        default_strides = read_launch_pixel_strides(af)
    except InvalidDatasetStructure:
        default_strides = DEFAULT_STRIDES

    # Overwrite pixel indexes for a given cycle number if necessary:
    keys = ("PixelBegin", "PixelEnd", "BinningSize")
    strides = {}
    for key in keys:
        stride = read(key, cycle_number)
        strides[key] = default_strides[key] if stride is None else stride

    return strides


def read_pixel_strides(af: "ArgoFloat") -> Dict[int, Dict[str, int]]:
    def read(pixel: Literal["PixelBegin", "PixelEnd", "BinningSize"]) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"
        values = [None for cyc in af.CYCLE_NUMBERS]

        if cfg_param in af.auxconfig:
            for ii, cyc in enumerate(af.CYCLE_NUMBERS):
                values[ii] = af.auxconfig.for_cycles(cfg_param, cyc)
        elif cfg_param in af.config:
            warnings.warn(
                f"Unexpectedly found {cfg_param} in the config parameters (not auxiliary)"
            )
            for ii, cyc in enumerate(af.CYCLE_NUMBERS):
                values[ii] = af.config.for_cycles(cfg_param, cyc)

        return values

    def describe_origin(strides) -> str:
        s = []
        for key, val in strides.items():
            if val is not None:
                s.append(key)
        s = ", ".join(s)

        return "default/launch config" if len(s) == 0 else f"mission specific ({s})"

    # Load default "launch" pixel indexes:
    try:
        default_strides = read_launch_pixel_strides(af)
    except InvalidDatasetStructure:
        default_strides = DEFAULT_STRIDES

    # Load pixel indexes for all cycles and overwrite default pixel indexes if necessary:
    strides = {}
    mission_strides = {}
    for ii, cyc in enumerate(af.CYCLE_NUMBERS):
        strides[cyc] = {}
        mission_strides[cyc] = {}

    keys = ("PixelBegin", "PixelEnd", "BinningSize")
    for key in keys:
        stride = read(key)  # Will return None if not specified
        for ii, cyc in enumerate(af.CYCLE_NUMBERS):
            mission_strides[cyc][key] = stride[
                ii
            ]  # Record mission strides to document origin below
            strides[cyc][key] = (
                default_strides[key] if stride[ii] is None else stride[ii]
            )

    for ii, cyc in enumerate(af.CYCLE_NUMBERS):
        strides[cyc]["origin"] = describe_origin(mission_strides[cyc])

    return strides
