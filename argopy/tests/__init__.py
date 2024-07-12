# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""
import argopy
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)
logging.getLogger("pyftpdlib").setLevel(logging.ERROR)
# logging.getLogger("MainThread").setLevel(logging.ERROR)
logging.getLogger("s3fs").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("aiobotocore").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)

# log = logging.getLogger("argopy.tests")

argopy.set_options(api_timeout=2 * 60)  # From Github actions, requests can take a while
# argopy.set_options(api_timeout=30)
argopy.show_options()
argopy.show_versions()
