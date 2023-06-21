Data sources
============

|Profile count| |Profile BGC count|

|Erddap status| |GDAC status| |Argovis status| |Statuspage|

.. |Erddap status| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_erddap.json
.. |GDAC status| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_gdac.json
.. |Argovis status| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_argovis.json
.. |Profile count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json
.. |Profile BGC count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20BGC%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-BGC.json
.. |Statuspage| image:: https://img.shields.io/static/v1?label=&message=Check%20all%20Argo%20monitors&color=blue&logo=statuspage&logoColor=white
   :target: https://argopy.statuspage.io

.. contents::
   :local:

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher as ArgoDataFetcher

Available data sources
----------------------

**argopy** can get access to Argo data from the following sources:

1. the `Ifremer erddap server <http://www.ifremer.fr/erddap>`__.
    The erddap server database is updated daily and doesn’t require you to download anymore data than what you need.
    You can select this data source with the keyword ``erddap`` and methods described below.
    The Ifremer erddap dataset is based on mono-profile files of the GDAC.
    Since this is the most efficient method to fetcher Argo data, it's the default data source in **argopy**.

2. an Argo GDAC server or any other GDAC-compliant folders.
    You can fetch data from any of the 3 official GDAC online servers: the Ifremer https and ftp and the US ftp.
    This data source can also point toward your own local copy of the `GDAC
    ftp content <http://www.argodatamgt.org/Access-to-data/Argo-GDAC-ftp-and-https-servers>`__.
    You can select this data source with the keyword ``gdac`` and methods described below.

3. the `Argovis server <https://argovis.colorado.edu/>`__.
    The Argovis server database is updated daily and only provides access to curated Argo data (QC=1 only).
    You can select this data source with the keyword ``argovis`` and methods described below.


Selecting a source
------------------

You have several ways to specify which data source you want to use:

-  **using argopy global options**:

.. ipython:: python
    :okwarning:

    argopy.set_options(src='erddap')

-  **in a temporary context**:

.. ipython:: python
    :okwarning:

    with argopy.set_options(src='erddap'):
        loader = ArgoDataFetcher().profile(6902746, 34)

-  **with an argument in the data fetcher**:

.. ipython:: python
    :okwarning:

    loader = ArgoDataFetcher(src='erddap').profile(6902746, 34)


Comparing data sources
----------------------

Features
~~~~~~~~

Each of the available data sources have their own features and
capabilities. Here is a summary:

======================= ====== ==== ===== =======
Data source:            erddap gdac local argovis
======================= ====== ==== ===== =======
**Access Points**
region                  X      X    X     X
float                   X      X    X     X
profile                 X      X    X     X
**User mode**
standard                X      X    X     X
expert                  X      X    X
research                X      X    X
**Dataset**
core (T/S)              X      X    X     X
BGC (experimental)      X      X    X
Reference data for DMQC X
Trajectories
**Parallel method**                     
multi-threading         X      X    X     X
multi-processes                     X
Dask client (experimental)
**Offline mode**                    X
======================= ====== ==== =======

Fetched data and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

| You may wonder if the fetched data are different from the available
  data sources.
| This will depend on the last update of each data sources and of your
  local data.

Let's retrieve one float data from a local sample of the GDAC ftp (a sample GDAC ftp is downloaded automatically with the method :meth:`argopy.tutorial.open_dataset`):

.. ipython:: python
    :okwarning:

    # Download ftp sample and get the ftp local path:
    ftproot = argopy.tutorial.open_dataset('gdac')[0]
    
    # then fetch data:
    with argopy.set_options(src='gdac', ftp=ftproot):
        ds = ArgoDataFetcher().float(1900857).load().data
        print(ds)

Let’s now retrieve the latest data for this float from the ``erddap`` and ``argovis`` sources:

.. ipython:: python
    :okwarning:

    with argopy.set_options(src='erddap'):
        ds = ArgoDataFetcher().float(1900857).load().data
        print(ds)

.. ipython:: python
    :okwarning:

    with argopy.set_options(src='argovis'):
        ds = ArgoDataFetcher().float(1900857).load().data
        print(ds)

We can see some minor differences between ``gdac``/``erddap`` vs the
``argovis`` response.

.. _api-status:

Status of sources
-----------------

With remote, online data sources, it may happens that the data server is experiencing down time. 
With local data sources, the availability of the path is checked when it is set. But it may happens that the path points to a disk that get unmounted or unplugged after the option setting.

If you're running your analysis on a Jupyter notebook, you can use the :meth:`argopy.status` method to insert a data status monitor on a cell output. All available data sources will be monitored continuously.

.. code-block:: python

    argopy.status()

.. image:: _static/status_monitor.png
  :width: 350
  
If one of the data source become unavailable, you will see the status bar changing to something like:
  
.. image:: _static/status_monitor_down.png
  :width: 350  
  
Note that the :meth:`argopy.status` method has a ``refresh`` option to let you specify the refresh rate in seconds of the monitoring.

Last, you can check out `the following argopy status webpage that monitors all important resources to the software <https://argopy.statuspage.io>`_.


Setting-up your own local copy of the GDAC ftp
----------------------------------------------

Data fetching with the ``gdac`` data source will require you to
specify the path toward your local copy of the GDAC ftp server with the
``ftp`` option.

This is not an issue for expert users, but standard users may wonder how
to set this up. The primary distribution point for Argo data, the only
one with full support from data centers and with nearly a 100% time
availability, is the GDAC ftp. Two mirror servers are available:

-  France Coriolis: ftp://ftp.ifremer.fr/ifremer/argo
-  US GODAE: ftp://usgodae.org/pub/outgoing/argo

If you want to get your own copy of the ftp server content, you have 2 options detailed below.


Copy with DOI reference
~~~~~~~~~~~~~~~~~~~~~~~

If you need an Argo database referenced with a DOI, one that you could use to make your analysis reproducible, then we
recommend you to visit https://doi.org/10.17882/42182. There, you will find links toward monthly snapshots of the
Argo database, and each snapshot has its own DOI.

For instance, https://doi.org/10.17882/42182#92121 points toward the snapshot archived on February 10st 2022. Simply
download the tar archive file (about 44Gb) and uncompress it locally.

You're done !

Synchronized copy
~~~~~~~~~~~~~~~~~

If you need a local Argo database always up to date with the GDAC server,
Ifremer provides a nice rsync service. The rsync server “vdmzrs.ifremer.fr”
provides a synchronization service between the “dac” directory of the
GDAC and a user mirror. The “dac” index files are also available from
“argo-index”.

From the user side, the rsync service:

-  Downloads the new files
-  Downloads the updated files
-  Removes the files that have been removed from the GDAC
-  Compresses/uncompresses the files during the transfer
-  Preserves the files creation/update dates
-  Lists all the files that have been transferred (easy to use for a
   user side post-processing)

To synchronize the whole dac directory of the Argo GDAC:

.. code:: bash

   rsync -avzh --delete vdmzrs.ifremer.fr::argo/ /home/mydirectory/...

To synchronize the index:

.. code:: bash

   rsync -avzh --delete vdmzrs.ifremer.fr::argo-index/ /home/mydirectory/...

.. note::

    The first synchronisation of the whole dac directory of the Argo GDAC (365Gb) can take quite a long time (several hours).

