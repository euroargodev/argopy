.. currentmodule:: argopy
.. _expert-tools-stores:

Advanced Argo file stores
=========================

Advanced Argo file stores provided by **argopy** are classes designed to help users to separate `Input/Output operations` from `data analysis` in their procedures.

More precisely, *Argo file stores* are any convenience classes providing an Argo specific and facilitated access to Argo files, with some file system logic and that are protocol agnostic.
For instance, if appropriate, these classes can provide methods like `open_dataset`, `open_json`, `open`, `ls`, `exists`, whatever the file type and location (local folder, http, ftp or s3).

By default, these Argo file stores are instantiated using the global option `gdac` pointing toward the desired data source (any valid GDAC host can be used). After instantiation, a data analysis procedure can be developed, agnostic of the file source and associated access protocol.

Note that given the rather low-level implementation of these API, Argo file stores are probably for users with some advanced knowledge of the Argo dataset: analysis on raw data, Argo index files manipulation, WMO oriented procedures, etc...

These classes are all explained in the following sections:

* :doc:`Store for float files <argofloat>`
* :doc:`Store for index files <argoindex>`
* :doc:`Store for GDAC files <gdac_filesystem>`

.. toctree::
    :maxdepth: 2
    :hidden:

    Store for float files <argofloat>
    Store for index files <argoindex>
    Store for GDAC files <gdac_filesystem>


In a nutshell
-------------

.. tab-set::

    .. tab-item:: Float files

        Working on a specific float ? get simplified access to any of its netcdf dataset:

        .. ipython:: python
            :okwarning:

            from argopy import ArgoFloat

            ds = ArgoFloat(6903091).open_dataset('meta')
            ds


    .. tab-item:: Index files

        Load, curate, search most of Argo index files:

        .. ipython:: python
            :okwarning:

            from argopy import ArgoIndex

            idx = ArgoIndex(index_file="bgc-s")
            idx.query.wmo(6903091)
            df = idx.to_dataframe()
            df

    .. tab-item:: GDAC files

        Provide a low-level access to a GDAC content, protocol agnostic, like a generic file system:

        .. ipython:: python
            :okwarning:

            from argopy import gdacfs

            fs = gdacfs("https://data-argo.ifremer.fr")
            fs.glob("dac/aoml/13857/profiles/*00*.nc")
