import numpy as np


def Z_euphotic(depth: np.array, PAR: np.array, max_surface: float = 5.) -> float:
    """Compute depth of the euphotic zone

    This is the depth at which the downwelling irradiance is reduced to 1% of its surface value.

    The surface value is taken as the maximum `PAR` above `max_surface`.

    The downwelling irradiance is from the photosynthetically available radiation: PAR.

    Parameters
    ----------
    depth: np.array
        Vertical axis values, pressure or depth.
    PAR: np.array
        Photosynthetically available radiation: PAR.
    max_surface: float, optional, default: 5.
        Maximum value of the vertical axis above which the maximum PAR value is considered surface values.

    Returns
    -------
    Euphotic layer depth estimate
    """
    idx = ~np.logical_or(np.isnan(depth), np.isnan(PAR))
    depth = depth[idx]
    PAR = PAR[idx]

    try:
        Surface_levels = np.where(depth <= max_surface)[0]
    except:
        Surface_levels = np.array()
        pass

    result = np.nan
    if Surface_levels.shape[0] > 0:
        Surface_value = np.max(PAR[Surface_levels])
        if np.any(PAR > (Surface_value / 100)):
            index_1_percent = np.argmin(np.abs(PAR - (Surface_value / 100)))
            result = depth[index_1_percent]

    return np.array(result)