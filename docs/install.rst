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

you can also work with the latest version:

.. code-block:: text

    pip install git+http://github.com/euroargodev/argopy.git@master


.. _bgc-install:

Install release with partial 🟢 **bgc** support
-----------------------------------------------

.. versionadded:: v0.1.14rc2

🟢 **bgc** support is provided as a **release candidate** only. Therefore, it is not available in conda and won't be selected by default with pip.

To install **argopy** with *partial* 🟢 **bgc** support, you need to use:

.. code-block:: text

    pip install argopy==0.1.14rc2


Required dependencies
---------------------

- aiohttp
- erddapy
- fsspec
- netCDF4
- scipy
- toolz
- xarray
- requests

Note that Erddapy_ is required because `erddap <https://coastwatch.pfeg.noaa.gov/erddap/information.html>`_ is the default data fetching backend.

Requirement dependencies details can be found `here <https://github.com/euroargodev/argopy/network/dependencies#requirements.txt>`_.

The **argopy** software is `continuously tested <https://github.com/euroargodev/argopy/actions?query=workflow%3Atests>`_ under latest OS (Linux, Mac OS and Windows) and with python versions 3.8 and 3.9

Optional dependencies
---------------------

For a complete **argopy** experience, you may also consider to install the following packages:

**Utilities**

- gsw
- tqdm
- zarr

**Performances**

- dask
- distributed
- pyarrow

**Visualisation**

- IPython
- cartopy
- ipykernel
- ipywidgets
- matplotlib
- seaborn



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
