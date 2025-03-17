Installation
============

|License| |Python version| |Anaconda-Server Badge|

|pypi dwn| |conda dwn|

Instructions
------------

Install the last release with conda:

.. code-block:: text

    conda install -c conda-forge argopy

or pip:

.. code-block:: text

    pip install argopy

you can also work with the ongoing development version:

.. code-block:: text

    pip install git+http://github.com/euroargodev/argopy.git@master


Required dependencies
---------------------

- xarray < 2024.3.0 : because of `this issue <https://github.com/pydata/xarray/issues/8909>`_ (see also :issue:`390` and :issue:`404`). As of March 2025, a fix is on the way at `xarray <https://github.com/pydata/xarray/pull/9273>`_ but not yet available.
- scipy
- numpy < 2 : because of the xarray limitation above
- erddapy
- netCDF4
- h5netcdf
- fsspec < 2025.3 : because of `this issue <https://github.com/euroargodev/argopy/issues/459>`_.
- toolz
- requests
- aiohttp
- decorator
- packaging

Requirement dependencies details can be found `here <https://github.com/euroargodev/argopy/blob/master/requirements.txt>`_.

The **argopy** software is `continuously tested <https://github.com/euroargodev/argopy/actions?query=workflow%3Atests>`_ under latest OS (Linux, Mac OS and Windows) and with python versions 3.10 and 3.11

Optional dependencies
---------------------

For a complete **argopy** experience, you may also consider to install the following packages:

**Utilities**

- gsw
- tqdm

**Performances**

- dask
- distributed
- pyarrow

**Files handling**

- boto3 / s3fs
- zarr
- numcodecs
- kerchunk

**Visualisation**

- cartopy
- IPython
- ipykernel
- ipywidgets
- matplotlib
- pyproj
- seaborn


Environments
------------

You can simply use one of the **argopy** testing conda environments listed here: https://github.com/euroargodev/argopy/tree/master/ci/requirements

You can also look at this section of the documentation :ref:`contributing.dev_envc`


.. _Erddapy: https://github.com/ioos/erddapy
.. |Gitter| image:: https://badges.gitter.im/Argo-floats/argopy.svg
   :target: https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |License| image:: https://img.shields.io/badge/License-EUPL%201.2-brightgreen
    :target: https://opensource.org/license/eupl-1-2/
.. |Python version| image:: https://img.shields.io/pypi/pyversions/argopy
   :target: //pypi.org/project/argopy/
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/argopy/badges/platforms.svg
   :target: https://anaconda.org/conda-forge/argopy
.. |pypi dwn| image:: https://img.shields.io/pypi/dm/argopy?label=Pypi%20downloads
   :target: //pypi.org/project/argopy/
.. |conda dwn| image:: https://img.shields.io/conda/dn/conda-forge/argopy?label=Conda%20downloads
   :target: //anaconda.org/conda-forge/argopy
.. |PyPI| image:: https://img.shields.io/pypi/v/argopy
   :target: //pypi.org/project/argopy/
.. |Conda| image:: https://anaconda.org/conda-forge/argopy/badges/version.svg
   :target: //anaconda.org/conda-forge/argopy
.. |tests in FREE env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-free.yml/badge.svg
.. |tests in DEV env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-dev.yml/badge.svg
.. |image20| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |image21| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |badge| image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge
   :target: https://mybinder.org/v2/gh/euroargodev/binder-sandbox/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Feuroargodev%252Fargopy%26urlpath%3Dlab%252Ftree%252Fargopy%252Fdocs%252Ftryit.ipynb%26branch%3Dmaster
