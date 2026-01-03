from typing import Any

from argopy.utils.format import cfgnameparser
from argopy.utils.casting import to_bool


def cast_config_parameter(param: str, pvalue: Any) -> float | int | bool:
    """Cast one float configuration parameter

    Parameters
    ----------
    param: str
        Name of the configuration parameter.
        Name is used to infer unit and dtype.
        Valid parameters are taken from R18.

    Returns
    -------
    float, int, bool
        Cast parameter value, according to parameter name inferred unit
    """
    if not param.startswith("CONFIG_"):
        param = f"CONFIG_{param}"
    if pvalue is None:
        return None
    unit = cfgnameparser(param)["unit"]
    if unit in [
        "angulardeg",
        "cbar",
        "cm/s",
        "cm^3",
        "csec",
        "dbar",
        "dd",
        "degc",
        "floatday",
        "hertz",
        "hh",
        "hhmm",
        "hours",
        "kbyte",
        "m^-1",
        "mdegc",
        "minutes",
        "mm",
        "mm/s",
        "mpsu",
        "msec",
        "nm",
        "seconds",
        "umol/l",
        "volts",
        "yyyy",
    ]:
        return float(pvalue)
    elif unit in ["count", "number", "days"]:
        return int(pvalue)
    elif unit in ["logical"]:
        return to_bool(pvalue)
    raise ValueError(
        f"Parameter '{param}' has an undocumented unit '{unit}'. If you think this is unjustified, please raise an issue at https://github.com/euroargodev/argopy/issues."
    )
