Installation
============

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- Python 3.6
- Xarray 0.14
- Erddapy 0.5
- Fsspec 0.7
- Gsw 3.3

Note that Erddapy_ is required because `erddap <https://coastwatch.pfeg.noaa.gov/erddap/information.html>`_ is the default data fetching backend.

The **argopy** software is continuously tested with sucess under all OS (Linux, Mac and Windows) and with python versions 3.6, 3.7 and 3.8.

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

