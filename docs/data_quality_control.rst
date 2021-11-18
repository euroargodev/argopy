.. _data_qc:


Data quality control
====================

**argopy** comes with handy methods to help you quality control measurements. This section is probably intended for `expert` users.

Most of these methods are available through the :class:`xarray.Dataset` accessor namespace ``argo``. This means that if your dataset is `ds`, then you can use `ds.argo` to access more **argopy** functionalities.

Let's start with import and set-up:

.. ipython:: python
    :okwarning:

    import os
    os.makedirs('float_source', exist_ok=True)

    from argopy import DataFetcher as ArgoDataFetcher


Salinity calibration
--------------------

The Argo salinity calibration method is called OWC_, after the names of the core developers: Breck Owens, Anny Wong and Cecile Cabanes.
Historically, the OWC method has been implemented in `Matlab <https://github.com/ArgoDMQC/matlab_owc>`_ . More recently a `python version has been developed <https://github.com/euroargodev/argodmqc_owc>`_. At this point, both software take as input a pre-processed version of the Argo float data to evaluate/calibrate.

**argopy** is able to perform this preprocessing and to create a *float source* data to be used by OWC software. This is made by :meth:`argopy.xarray.ArgoAccessor.create_float_source`.

First, you would need to fetch the Argo float data you want to calibrate, in `expert` mode:

.. ipython:: python
    :okwarning:

    ds = ArgoDataFetcher(mode='expert').float(6902766).load().data

Then, to create the float source data is as simple as:

.. ipython:: python
    :okwarning:

    ds.argo.create_float_source("float_source")

This will create the "float_source/6902766.mat" Matlab files to be set directly in the configuration file of the OWC software. This routine implements the same pre-processing as in the Matlab version (which is hosted on `this repo <https://github.com/euroargodev/dm_floats>`_ and ran with `this routine <https://github.com/euroargodev/dm_floats/blob/master/src/ow_source/create_float_source.m>`_).

.. note::
    If the dataset contains data from more than one float, several Matlab files are created, one for each float. This will allow you to prepare data from a collection of floats more easily.

If you don't specify a path name, the method returns a dictionary with the float WMO as keys and pre-processed data (as :class:`xarray.Dataset`) as values.

.. ipython:: python
    :okwarning:

    ds_source = ds.argo.create_float_source()
    ds_source

See all options available for this method here: :meth:`argopy.xarray.ArgoAccessor.create_float_source`.

**argopy** also provides an OWC variables filter named :meth:`argopy.xarray.ArgoAccessor.filter_scalib_pres`. This method allows you to filter variables according to OWC salinity calibration software requirements. This filter modifies pressure, temperature and salinity related variables of the dataset.

.. [OWC] See all the details about the OWC methodology in these references:
"An improved calibration method for the drift of the conductivity sensor on autonomous CTD profiling floats by θ–S climatology".
Deep-Sea Research Part I: Oceanographic Research Papers, 56(3), 450-457, 2009. https://doi.org/10.1016/j.dsr.2008.09.008

"Improvement of bias detection in Argo float conductivity sensors and its application in the North Atlantic".
Deep-Sea Research Part I: Oceanographic Research Papers, 114, 128-136, 2016. https://doi.org/10.1016/j.dsr.2016.05.007

