.. currentmodule:: xarray

Salinity calibration
--------------------

The Argo salinity calibration method is called [OWC]_, after the names of the core developers: Breck Owens, Anny Wong and Cecile Cabanes. Historically, the OWC method has been implemented in `Matlab <https://github.com/ArgoDMQC/matlab_owc>`_ . More recently a `python version has been developed <https://github.com/euroargodev/argodmqc_owc>`_.

Preprocessing data
^^^^^^^^^^^^^^^^^^

At this point, both OWC software take as input a pre-processed version of the Argo float data to evaluate/calibrate.

**argopy** is able to perform this preprocessing and to create a *float source* data to be used by OWC software. This is made by :meth:`Dataset.argo.create_float_source`.

First, you would need to fetch the Argo float data you want to calibrate, in ``expert`` mode:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher

    ds = DataFetcher(mode='expert').float(6902766).load().data

Then, to create the float source data, you call the method and provide a folder name to save output files:

.. ipython:: python
    :okwarning:

    ds.argo.create_float_source("float_source")

This will create the ``float_source/6902766.mat`` Matlab files to be set directly in the configuration file of the OWC software. This routine implements the same pre-processing as in the Matlab version (which is hosted on `this repo <https://github.com/euroargodev/dm_floats>`_ and ran with `this routine <https://github.com/euroargodev/dm_floats/blob/master/src/ow_source/create_float_source.m>`_). All the detailed steps of this pre-processing are given in the :meth:`Dataset.argo.create_float_source` API page.

.. note::
    If the dataset contains data from more than one float, several Matlab files are created, one for each float. This will allow you to prepare data from a collection of floats.

If you don't specify a path name, the method returns a dictionary with the float WMO as keys and pre-processed data as :class:`xarray.Dataset` as values.

.. ipython:: python
    :okwarning:

    ds_source = ds.argo.create_float_source()
    ds_source

See all options available for this method here: :meth:`Dataset.argo.create_float_source`.

The method partially relies on two others:

- :meth:`Dataset.argo.filter_scalib_pres`: to filter variables according to OWC salinity calibration software requirements. This filter modifies pressure, temperature and salinity related variables of the dataset.

- :meth:`Dataset.argo.groupby_pressure_bins`: to sub-sampled measurements by pressure bins. This is an excellent alternative to the :meth:`Dataset.argo.interp_std_levels` to avoid interpolation and preserve values of raw measurements while at the same time aligning measurements along approximately similar pressure levels (depending on the size of the bins). See more description at here: :ref:`Pressure levels: Group-by bins`.

Running the calibration
^^^^^^^^^^^^^^^^^^^^^^^

Please refer to the `OWC python software documentation <https://github.com/euroargodev/argodmqc_owc>`_.

.. literalinclude:: owc_workflow_eg.py
    :language: python
    :caption: Typical OWC workflow example

OWC references
^^^^^^^^^^^^^^

.. [OWC] See all the details about the OWC methodology in these references:

"An improved calibration method for the drift of the conductivity sensor on autonomous CTD profiling floats by θ–S climatology".
Deep-Sea Research Part I: Oceanographic Research Papers, 56(3), 450-457, 2009. https://doi.org/10.1016/j.dsr.2008.09.008

"Improvement of bias detection in Argo float conductivity sensors and its application in the North Atlantic".
Deep-Sea Research Part I: Oceanographic Research Papers, 114, 128-136, 2016. https://doi.org/10.1016/j.dsr.2016.05.007
