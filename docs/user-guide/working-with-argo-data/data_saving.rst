Saving data
===========

.. contents::
   :local:

.. currentmodule:: xarray

Once you have your Argo data as a :class:`xarray.Dataset`, you can simply use the awesome export possibilities of `xarray <http://xarray.pydata.org>`_ like :meth:`xarray.Dataset.to_netcdf`.

But in doing so, you will produce a netcdf that is not compliant with the Argo CF convention.

**It is on our roadmap to provide the capability to save data in netcdf following the Argo CF convention.**

In the mean time, note that we provide a dedicated method to export an Argo :class:`xarray.Dataset` to zarr that will handle data type casting and compression easily, the :meth:`Dataset.argo.to_zarr` method:

.. code-block:: python
    :caption: Dataset export to zarr

    from argopy import DataFetcher
    ds = DataFetcher(src='gdac').float(6903091).to_xarray()
    # then:
    ds.argo.to_zarr("6903091_prof.zarr")
    # or:
    ds.argo.to_zarr("s3://argopy/sample-data/6903091_prof.zarr")
