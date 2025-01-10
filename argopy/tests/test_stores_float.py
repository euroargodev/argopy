import os

import pytest
import tempfile

import numpy as np
import pandas as pd
import importlib
import shutil
import logging
from urllib.parse import urlparse

import argopy
from argopy.errors import (
    GdacPathError,
    OptionValueError,
    InvalidDatasetStructure,
)
from argopy.utils.checkers import is_list_of_strings

from argopy.stores.float.implementations.argo_float_offline import ArgoFloatOffline
from argopy.stores.float.implementations.argo_float_online import ArgoFloatOnline

from mocked_http import mocked_httpserver, mocked_server_address

log = logging.getLogger("argopy.tests.floatstore")

skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")
skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")
