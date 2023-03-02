# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""
import argopy
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)
# log = logging.getLogger("argopy.tests")

argopy.set_options(api_timeout=4 * 60)  # From Github actions, requests can take a while
argopy.show_options()
argopy.show_versions()
