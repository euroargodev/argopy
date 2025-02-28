.. currentmodule:: argopy

.. _lazy:

Lazy dataset access
-------------------

.. warning::
    As of February 2025, this feature is considered experimental and can change without any deprecation warnings from
    one release to another. This is part of a wider effort to prepare **argopy** for evolutions of the Argo dataset in
    the cloud (cf the `ADMT working group on Argo cloud format activities <https://github.com/OneArgo/ADMT/issues/5>`_).

This **argopy** feature is implemented with ``open_dataset`` methods from: compatible argopy file stores (local, http, ftp and s3)
and the :class:`ArgoFloat` class. Since this is somehow a low-level implementation whereby users need to work with float data
directly, it is probably targetting users with operator or expert knowledge of Argo.

Contrary to the other performance improvement methods, this one is not accessible with a :class:`DataFetcher`.

What is it ?
~~~~~~~~~~~~

Lazyness in our use case, relates to limiting data transfert/load to what is really needed for an operation. For instance:

- if you want to work with a single Argo parameter for a given float, you don't need to download from the GDAC server all the other parameters,
- if you only are interested in assessing a file content (e.g. number of profiles or vertical levels), you also don't need to load anything else than the dimensions of the netcdf files.

Since a regular Argo netcdf is not intendeed to be accessed partially, it is rather tricky to access Argo data lazily.
Hopefully, the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library has been developped precisely for this use-case.

.. warning::
    Since variable content is not loaded, one limitation of the lazy approach, is that variables are not necessarily
    cast appropriately, and are often returned as simple *object*.

Compatible data sources
~~~~~~~~~~~~~~~~~~~~~~~

Not all file access protocols and servers support the byte range request that is mandatory to access lazilly a netcdf file.
The table below synthesises lazy support status for all possible GDAC hosts:

.. list-table:: GDAC hosts supporting lazy access to float netcdf dataset
    :header-rows: 1
    :stub-columns: 1

    * -
      - Supported
    * - https://data-argo.ifremer.fr
      - ❌
    * - https://usgodae.org/pub/outgoing/argo
      - ✅
    * - ftp://ftp.ifremer.fr/ifremer/argo
      - ✅
    * - s3://argo-gdac-sandbox/pub
      - ✅
    * - a local GDAC copy
      - ✅


**argopy** kerchunk helper
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to access lazily an Argo netcdf files, localy or remotely with a server supporting range requests, the netcdf
content has to be curated in order to make a byte range *catalogue* of its content. To do so, you need to have the
`kerchunk <https://fsspec.github.io/kerchunk/>`_ library installed in your working environment.

`kerchunk <https://fsspec.github.io/kerchunk/>`_ will analyse a netcdf file content (e.g. dimensions, list of variables)
and store these metadata in a json file compatible with zarr. With a specific syntax, these metadata can then be given
to :class:`xarray.open_dataset` to open a netcdf file lazily.

We developped a specific class to make this process easy for **argopy** users: :class:`stores.ArgoKerchunker`.

A typical use case will be to curate one or a collection of netcdf files and then save byte range *catalogues* in a specific store.

This can be done on the fly, but it can be time consuming, depending on the size of Argo netcdf datasets (the overhead
of using kerchunk may not be worth compared to download time). On the other hand, it may be interesting to save kerchunk data in a
shared store (local or remote), so that other users will be able to use it. From the user perspective, this has the huge
advantage of not requiring the kerchunk library anymore, since opening lazily a dataset will be done with the zarr engine with xarray.

This is demonstrated below:

.. ipython:: python
    :okwarning:

    from argopy.stores import ArgoKerchunker

    # Create an instance that will save netcdf byte range *catalogues* on a local folder:
    ak = ArgoKerchunker(store='local', root='~/myshared_kerchunk_data_folder')

    # Let's consider a remote Argo netcdf file from a server supporting lazy access:
    ncfile = "https://usgodae.org/pub/outgoing/argo/dac/coriolis/6903090/6903090_prof.nc"

    # Compute and save this netcdf byte range *catalogue* for later lazy access:
    js = ak.to_kerchunk(ncfile)


Now, for any user with read access to the `~/myshared_kerchunk_data_folder` folder, lazy access is possible without kerchunk:

.. ipython:: python
    :okwarning:

    from argopy.stores import httpstore

    # Create an instance where to find netcdf byte range *catalogues*:
    ak = ArgoKerchunker(store='local', root='~/myshared_kerchunk_data_folder')

    # Simply open the netcdf file lazily, giving the appropriate ArgoKerchunker:
    httpstore().open_dataset(ncfile, lazy=True, ak=ak)

.. _lazy-argofloat:

Laziness with an :class:`ArgoFloat`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When opening an Argo dataset from the :class:`ArgoFloat`, you can simply add the `lazy` argument to apply laziness:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat

    ds = ArgoFloat(6903091, host='us-http').open_dataset('prof', lazy=True)

without additional argument, the kerchunk store will be located in memory and the netcdf byte range *catalogues* computed on the fly using the kerchunk library.

If you want to specify a kerchunk data store, you can provide a :class:`stores.ArgoKerchunker` instance:

.. ipython:: python
    :okwarning:

    ak = ArgoKerchunker(store='local', root='~/myshared_kerchunk_data_folder')
    ds = ArgoFloat(6903091, host='us-http').open_dataset('prof', lazy=True, ak=ak)


In this scenario, if the dataset has already been processed by the :class:`stores.ArgoKerchunker` instance, the kerchunk library is not required to load lazily the dataset.

If the dataset has not been processed, then the kerchunk library is required.
