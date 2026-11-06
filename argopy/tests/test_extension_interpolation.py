import pytest
import logging
import numpy as np

from argopy.utils.computers import _std_level_boxes
from argopy.extensions.interpolation import StandardLevels

# TODO: move this test to test_utils_computers.py
def test_std_level_boxes():
    """Test box edges for standard levels"""
    levels = StandardLevels.from_product("EasyOneArgoLite").value
    lo, hi = _std_level_boxes(levels)

    assert (lo[0], hi[0]) == (0.5, 3.5)  # 2dbar
    assert (lo[-1], hi[-1]) == (5900.0, 6100.0)  # 6000dbar

