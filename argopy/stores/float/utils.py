from typing import Dict, Literal
from argopy.errors import InvalidDatasetStructure


def read_ramses_default_pixel_strides(af: "ArgoFloat") -> Dict[str, int]:
    def read(pixel: Literal['PixelBegin', 'PixelEnd', 'BinningSize']) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"

        if cfg_param in af.auxlaunchconfig:
            pixel_value = af.auxlaunchconfig[cfg_param]
        elif cfg_param in af.launchconfig:
            print(f"Unexpectedly found {cfg_param} in the launch config parameters (not auxiliary)")
            pixel_value = af.launchconfig[cfg_param]
        else:
            raise InvalidDatasetStructure(
                f"'{cfg_param}' cannot be found in the meta-aux (nore meta) configuration parameters")

        return pixel_value

    # Load "launch" pixel indexes (not mission dependents):
    strides = {
        'PixelBegin': read('PixelBegin'),
        'PixelEnd': read('PixelEnd'),
        'BinningSize': read('BinningSize'),
    }

    return strides


def read_ramses_pixel_strides_by_cycle(af: "ArgoFloat", cycle_number) -> Dict[str, int]:
    def read(pixel: Literal['PixelBegin', 'PixelEnd', 'BinningSize'], cycle) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"

        if cfg_param in af.auxconfig:
            pixel_value = af.auxconfig.for_cycles(cfg_param, cycle)
        elif cfg_param in af.config:
            print(f"Unexpectedly found {cfg_param} in the config parameters (not auxiliary)")
            pixel_value = af.config.for_cycles(cfg_param, cycle)
        else:
            return None

        return pixel_value

    # Load default "launch" pixel indexes:
    default_strides = read_ramses_default_pixel_strides(af)

    # Overwrite pixel indexes for a given cycle number if necessary:
    keys = ('PixelBegin', 'PixelEnd', 'BinningSize')
    strides = {}
    for key in keys:
        stride = read(key, cycle_number)
        strides[key] = default_strides[key] if stride is None else stride

    return strides


def read_ramses_pixel_strides(af: "ArgoFloat") -> Dict[int, Dict[str, int]]:
    def read(pixel: Literal['PixelBegin', 'PixelEnd', 'BinningSize']) -> int:
        cfg_param = f"CONFIG_RamsesAccOutput{pixel}_NUMBER"
        values = [None for cyc in af.CYCLE_NUMBERS]

        if cfg_param in af.auxconfig:
            for ii, cyc in enumerate(af.CYCLE_NUMBERS):
                values[ii] = af.auxconfig.for_cycles(cfg_param, cyc)
        elif cfg_param in af.config:
            print(f"Unexpectedly found {cfg_param} in the config parameters (not auxiliary)")
            for ii, cyc in enumerate(af.CYCLE_NUMBERS):
                values[ii] = af.config.for_cycles(cfg_param, cyc)

        return values

    # Load default "launch" pixel indexes:
    default_strides = read_ramses_default_pixel_strides(af)

    # Overwrite pixel indexes for a given cycle number if necessary:
    strides = {}
    for ii, cyc in enumerate(af.CYCLE_NUMBERS):
        strides[cyc] = {}

    keys = ('PixelBegin', 'PixelEnd', 'BinningSize')
    for key in keys:
        stride = read(key)
        for ii, cyc in enumerate(af.CYCLE_NUMBERS):
            strides[cyc][key] = default_strides[key] if stride[ii] is None else stride[ii]

    return strides

