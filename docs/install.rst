Installation
============

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- xarray>=0.15.1
- fsspec>=0.7.4, <0.8.4
- scikit-learn>=0.21
- erddapy>=0.6
- gsw>=3.3.1
- aiohttp>=3.6.2
- netCDF4>=1.3.1
- dask>=2.9
- toolz>=0.8.2

Note that Erddapy_ is required because `erddap <https://coastwatch.pfeg.noaa.gov/erddap/information.html>`_ is the default data fetching backend.

Requirement dependencies details can be found `here <https://github.com/euroargodev/argopy/network/dependencies#requirements.txt>`_.

The **argopy** software is `continuously tested <https://github.com/euroargodev/argopy/actions?query=workflow%3Atests>`_ with under latest OS (Linux and Mac OS) and with python versions 3.6, 3.7 and 3.8.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

For full **argopy** functionalities to work, the following packages are also required:

- ipython>=5.0.0
- ipywidgets>=7.5.1
- tqdm>=4.46.0
- Matplotlib 3.0
- Cartopy 0.17
- Seaborn 0.9.0

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

