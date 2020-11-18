Installation
============

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- Python
- Xarray
- Erddapy
- Fsspec
- Gsw

Note that Erddapy_ is required because `erddap <https://coastwatch.pfeg.noaa.gov/erddap/information.html>`_ is the default data fetching backend.

The **argopy** software is continuously tested with success under latest OS (Linux and Mac OS) and with python versions 3.6, 3.7 and 3.8.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

For full plotting functionality the following packages are required:

- Matplotlib 3.0 (mandatory)
- Cartopy 0.17 (for some methods only)
- Seaborn 0.9.0 (for some methods only)

Instructions
^^^^^^^^^^^^

Install the last release with conda:

.. code-block:: text

    conda install -c conda-forge argopy

or pip:

.. code-block:: text

    pip install argopy

you can also work with the latest version:

.. code-block:: text

    pip install git+http://github.com/euroargodev/argopy.git@master

.. _Erddapy: https://github.com/ioos/erddapy

