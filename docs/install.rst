Installation
============

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- Python 3.6
- Xarray 0.14
- Erddapy 0.5

Note that Erddapy_ is required because `erddap <https://coastwatch.pfeg.noaa.gov/erddap/information.html>`_ is the default data fetching backend.

The :mod:`argopy` library should work under all OS (Linux, Mac and Windows) and with python versions 3.6, 3.7 and 3.8.

For full plotting functionality the following packages are required:

- Matplotlib 3.0 (mandatory)
- Cartopy 0.17 (for some methods only)
- Seaborn 0.9.0 (for some methods only)

Instructions
^^^^^^^^^^^^

For the latest version:

.. code-block:: text

    pip install git+http://github.com/euroargodev/argopy.git@master

.. _Erddapy: https://github.com/ioos/erddapy

