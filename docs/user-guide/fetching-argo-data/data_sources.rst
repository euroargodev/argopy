.. currentmodule:: argopy
.. _data-sources:

Data sources
============

|Erddap status| |GDAC status| |Argovis status| |Statuspage|

.. |Erddap status| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Feuroargodev%2Fargopy-status%2Fmaster%2Fargopy_api_status_erddap.json
.. |GDAC status| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Feuroargodev%2Fargopy-status%2Fmaster%2Fargopy_api_status_gdac.json
.. |Argovis status| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Feuroargodev%2Fargopy-status%2Fmaster%2Fargopy_api_status_argovis.json
.. |Statuspage| image:: https://img.shields.io/static/v1?label=&message=Check%20all%20Argo%20monitors&color=blue&logo=statuspage&logoColor=white
   :target: https://argopy.statuspage.io

.. hint::

    **argopy** can fetch data from several data sources. To make sure you understand where you're getting data from, have a look at this section.

.. contents:: Contents
   :local:

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher as ArgoDataFetcher
    argopy.reset_options()

Available data sources
----------------------

**argopy** can get access to Argo data from the following sources:

1. ⭐ the `Ifremer erddap server <http://www.ifremer.fr/erddap>`__ (Default).
    The erddap server database is updated daily and doesn’t require you to download anymore data than what you need.
    You can select this data source with the keyword ``erddap`` and methods described below.
    The Ifremer erddap dataset is based on mono-profile files of the GDAC.
    Since this is the most efficient method to fetcher Argo data, it's the default data source in **argopy**.

2. 🌐 an Argo GDAC server or any other GDAC-compliant local folder.
    You can fetch data from any of the 3 official GDAC online servers: the Ifremer https and ftp and the US https.
    This data source can also point toward your own local copy of a `GDAC
    server content <http://www.argodatamgt.org/Access-to-data/Argo-GDAC-ftp-https-and-s3-servers>`__. You can list
    all available GDAC servers from :meth:`utils.list_gdac_servers` and you can select this data source with the keyword
    ``gdac`` and methods described below.

3. 👁 the `Argovis server <https://argovis.colorado.edu/>`__.
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

Each of the data sources have their own features and capabilities. In the table are summarized the **argopy** support for each data sources. Note that an unchecked dataset does not mean that it is not available in the data source, only that **argopy** does not support it, yet.

.. list-table:: Table of **argopy** data sources features
    :header-rows: 1
    :stub-columns: 2

    * -
      -
      - ``erddap``
      - ``gdac``
      - ``argovis``
    * -
      -
      - ⭐
      - 🌐
      - 👁
    * - :ref:`Access Points: <data-selection>`
      -
      -
      -
      -
    * -
      - 🗺 :ref:`region <data-selection-region>`
      - X
      - X
      - X
    * -
      - 🤖 :ref:`float <data-selection-float>`
      - X
      - X
      - X
    * -
      - ⚓ :ref:`profile <data-selection-profile>`
      - X
      - X
      - X
    * - :ref:`User mode: <user-mode-details>`
      -
      -
      -
      -
    * -
      - 🏄 expert
      - X
      - X
      -
    * -
      - 🏊 standard
      - X
      - X
      - X
    * -
      - 🚣 research
      - X
      - X
      -
    * - :ref:`Dataset: <data-set>`
      -
      -
      -
      -
    * -
      - core (T/S)
      - X
      - X
      - X
    * -
      - BGC
      - X
      -
      -
    * -
      - Deep
      - X
      - X
      - X
    * -
      - Trajectories
      -
      -
      -
    * -
      - Reference data for DMQC
      - X
      -
      -


Fetched data and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

| You may wonder if the fetched data are different from the available
  data sources.
| This will depend on the last update of each data sources and of your
  local data.

.. tabs::

    .. tab:: **GDAC** servers

        Let's retrieve one float data from a local sample of the GDAC (a GDAC sample can be downloaded automatically with the method :meth:`argopy.tutorial.open_dataset`):

        .. ipython:: python
            :okwarning:

            # Download a GDAC sample and get the corresponding local path:
            gdacroot = argopy.tutorial.open_dataset('gdac')[0]

            # then fetch data:
            with argopy.set_options(src='gdac', gdac=gdacroot):
                ds = ArgoDataFetcher().float(1900857).load().data
                print(ds)

    .. tab:: **erddap** API

        Let’s now retrieve the latest data for this float from the ``erddap``:

        .. ipython:: python
            :okwarning:

            with argopy.set_options(src='erddap'):
                ds = ArgoDataFetcher().float(1900857).load().data
                print(ds)

    .. tab:: **argovis** API

        And with ``argovis``:

        .. ipython:: python
            :okwarning:

            with argopy.set_options(src='argovis'):
                ds = ArgoDataFetcher().float(1900857).load().data
                print(ds)


.. _api-status:

Status of sources
-----------------

With remote, online data sources, it may happens that the data server is experiencing down time. 
With local data sources, the availability of the path is checked when it is set. But it may happens that the path points to a disk that get unmounted or unplugged after the option setting.

If you're running your analysis on a Jupyter notebook, you can use the :meth:`argopy.status` method to insert a data status monitor on a cell output. All available data sources will be monitored continuously.

.. code-block:: python

    argopy.status()

.. image:: ../../_static/status_monitor.png
  :width: 350
  
If one of the data source become unavailable, you will see the status bar changing to something like:
  
.. image:: ../../_static/status_monitor_down.png
  :width: 350  
  
Note that the :meth:`argopy.status` method has a ``refresh`` option to let you specify the refresh rate in seconds of the monitoring.

Last, you can check out `the following argopy status webpage that monitors all important resources to the software <https://argopy.statuspage.io>`_.


Setting-up your own local copy of the GDAC https server
-------------------------------------------------------

Data fetching with the ``gdac`` data source will require you to
specify the path toward your local copy of the GDAC server with the
``gdac`` option.

This is not an issue for expert users, but standard users may wonder how
to set this up. The primary distribution point for Argo data, the only
one with full support from data centers and with nearly a 100% time
availability, is a GDAC https server. Two servers are available:

-  France Coriolis: https://data-argo.ifremer.fr
-  USA GODAE: https://usgodae.org/pub/outgoing/argo

If you want to get your own copy of the GDAC server content, you have 2 options detailed below.


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

