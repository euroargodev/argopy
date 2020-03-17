#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 12/03/2020


import requests
import io
from IPython.core.display import display, HTML

def urlopen(url):
    """ Load content from url or raise alarm on status with explicit information on the error

    """
    # https://github.com/ioos/erddapy/blob/3828a4f479e7f7653fb5fd78cbce8f3b51bd0661/erddapy/utilities.py#L37
    r = requests.get(url)
    data = io.BytesIO(r.content)

    if r.status_code == 200:  # OK
        return data

    # 4XX client error response
    elif r.status_code == 404:  # Empty response
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8").replace("Error", ""))
        error.append("%s" % url)
        raise requests.HTTPError("\n".join(error))

    # 5XX server error response
    elif r.status_code == 500:  # 500 Internal Server Error
        if "text/html" in r.headers.get('content-type'):
            display(HTML(data.read().decode("utf-8")))
        error = ["Error %i " % r.status_code]
        error.append(data.decode("utf-8"))
        error.append("%s" % url)
        raise requests.HTTPError("\n".join(error))
    else:
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8"))
        error.append("%s" % url)
        print("\n".join(error))
        r.raise_for_status()
