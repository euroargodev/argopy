.. currentmodule:: argopy

.. _lazy:

Lazy dataset access
-------------------

.. warning::
    As of February 2025, this feature is considered highly experimental so that there is no guarantee that it is fully functional, documentation may be broke and API can change without any deprecation warnings from one release to another. This is part of a wider effort to prepare **argopy** for evolutions of the Argo dataset in
    the cloud (cf the `ADMT working group on Argo cloud format activities <https://github.com/OneArgo/ADMT/issues/5>`_).

This **argopy** feature is implemented with the ``open_dataset`` methods relying on the s3 store: :class:`stores.s3store` and
and :class:`ArgoFloat`. Since this is somehow a low-level implementation whereby users need to work with float data
directly, it is probably targeting users with operator or expert knowledge of Argo.

Contrary to the other performance improvement methods, this one is not accessible with a :class:`DataFetcher`.

What is laziness ?
~~~~~~~~~~~~~~~~~~

Laziness in our use case, relates to limiting remote data transfer/load to what is really needed for an operation. For instance:

- if you want to work with a single Argo parameter for a given float, you don't need to download from the GDAC server all the other parameters,
- if you only are interested in assessing a file content (e.g. number of profiles or vertical levels), you also don't need to load anything else than the dimensions of the netcdf files.

Remote laziness is natively supported by some file formats like `zarr <https://zarr.dev/>`_ and `parquet <https://parquet.apache.org/>`_. However, netcdf was not designed for this use case.

How it works ?
~~~~~~~~~~~~~~

Since a regular Argo netcdf is not intended to be accessed partially from a remote server, it is rather tricky to access Argo data lazily.
Hopefully, the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library has been developed precisely for this use-case.

In order to access lazily a remote Argo netcdf file with a server supporting byte range requests, the netcdf
content has to be analysed in order to make a byte range *catalogue* of its content: this is called a *reference*.
To do so, you need to have the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library installed in your working environment.

If someone or some party has created the netcdf *reference* and makes it accessible, you don't need the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library installed.

`kerchunk <https://fsspec.github.io/kerchunk/>`_ will analyse a netcdf file content (e.g. dimensions, list of variables)
and store these metadata in a json file compatible with zarr. With a specific syntax, these metadata can then be given
to :func:`xarray.open_dataset` or :func:`xarray.open_zarr` to open a netcdf file lazily.


.. currentmodule:: xarray

.. warning::
    Since variable content is not loaded, one limitation of the lazy approach, is that variables are not necessarily
    cast appropriately, and are often returned as simple *object*.

    You can use the :func:`Dataset.argo.cast_types` method to cast Argo variables correctly.

.. currentmodule:: argopy


Laziness support status
~~~~~~~~~~~~~~~~~~~~~~~

Not all Argo data servers support the byte range request that is mandatory to access lazily a netcdf file; and not all **argopy** methods support laziness through kerchunk and zarr references data.

The table below syntheses lazy support status for all possible GDAC hosts:

.. list-table:: Laziness support status for **argopy** users
    :header-rows: 1
    :stub-columns: 1

    * - GDAC hosts
      - Server support
      - **argopy** support
      - Can you use it ?
    * - https://data-argo.ifremer.fr
      - ❌
      - ❌
      - ❌
    * - https://usgodae.org/pub/outgoing/argo
      - ✅
      - 🛠
      - ❌
    * - ftp://ftp.ifremer.fr/ifremer/argo
      - ✅
      - ✅
      - ✅
    * - `s3://argo-gdac-sandbox/pub <https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/index.html#pub/>`_
      - ✅
      - ✅
      - ✅
    * - a local GDAC copy
      - ✅
      - ✅
      - ❌


.. _lazy-argofloat:

Laziness with an :class:`ArgoFloat` or :class:`gdacfs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When opening an Argo dataset with :class:`ArgoFloat`, you can simply add the `lazy` argument to apply laziness:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat

    ds = ArgoFloat(6903091, host='s3').open_dataset('prof', lazy=True)

without additional argument, netcdf reference data are computed on the fly using the kerchunk library and stored in memory.

You can also use laziness from a :class:`gdacfs`:

.. ipython:: python
    :okwarning:

    from argopy import gdacfs

    ds = gdacfs('s3').open_dataset("dac/coriolis/6903090/6903090_prof.nc", lazy=True)

A simple way to ensure that you open the dataset lazily is to check for the source value in the encoding attribute, it should be:

.. ipython:: python
    :okwarning:

    ds.encoding['source'] == 'reference://'


**argopy** kerchunk helper
~~~~~~~~~~~~~~~~~~~~~~~~~~

Under the hood, :class:`ArgoFloat` and :class:`gdacfs` will rely on kerchunk reference data provided by a
:class:`stores.ArgoKerchunker` instance.

A typical direct use case of :class:`stores.ArgoKerchunker` is to save kerchunk data in a shared store (local or remote),
so that other users will be able to use it. From the user perspective, this has the huge advantage of not requiring the
kerchunk library, since opening lazily a dataset will be done with the zarr engine of xarray.

It could go like this:

.. ipython:: python
    :okwarning:

    from argopy.stores import ArgoKerchunker

    # Create an instance that will save netcdf to zarr references on a local
    # folder at "~/kerchunk_data_folder":
    ak = ArgoKerchunker(store='local',
                        root='~/kerchunk_data_folder',
                        storage_options={'anon': True}, # so that AWS credentials are not required
                        )

    # Note that you could also use a remote reference store, for instance:
    #ak = ArgoKerchunker(store=fsspec.filesystem('dir',
    #                                            path='s3://.../kerchunk_data_folder/',
    #                                            target_protocol='s3'))

Now we can get a dummy list of netcdf files:

.. ipython:: python
    :okwarning:

    from argopy import ArgoIndex
    idx = ArgoIndex(host='s3').query.box([-65, -55, 30, 40,
                                          '2025-01-01', '2025-02-01'])

    ncfiles = [af.ls_dataset()['prof'] for af in idx.iterfloats()]
    print(len(ncfiles))

and compute zarr references that will be saved by the :class:`stores.ArgoKerchunker` instance. Note that this computation is done using Dask delayed when available, otherwise using multithreading:

.. ipython:: python
    :okwarning:

    ak.translate(ncfiles, fs=idx.fs['src'], chunker='auto');

The ``chunker`` option determines which chunker to use, which is different for netcdf 3 and netcdf4/hdf5 files. Checkout the API documentation for more details here :meth:`stores.ArgoKerchunker.translate`.

To later reuse such references to open lazily one of these netcdf files, an operation that does not require the
`kerchunk <https://fsspec.github.io/kerchunk/>`_ library, you can provide the appropriate :class:`stores.ArgoKerchunker`
instance to a :class:`ArgoFloat` or :class:`gdacfs`:

.. ipython:: python
    :okwarning:

    ak = ArgoKerchunker(store='local',
                        root='~/kerchunk_data_folder',
                        storage_options={'anon': True}, # so that AWS credentials are not required
                        )

    wmo = idx.read_wmo()[0]  # Select one float from the index search above

    ds = ArgoFloat(wmo, host='s3').open_dataset('prof', lazy=True, ak=ak)

    ds.encoding['source'] == 'reference://'
