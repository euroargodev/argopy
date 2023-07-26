.. _data_qc:

Data quality control
====================

.. contents::
   :local:

**argopy** comes with methods to help you quality control measurements. This section is probably intended for `expert` users.

Most of these methods are available through the :class:`xarray.Dataset` accessor namespace ``argo``. This means that if your dataset is `ds`, then you can use `ds.argo` to access more **argopy** functionalities.

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    argopy.clear_cache()
    argopy.reset_options()
    from argopy import DataFetcher


Salinity calibration
--------------------

.. currentmodule:: xarray

The Argo salinity calibration method is called [OWC]_, after the names of the core developers: Breck Owens, Anny Wong and Cecile Cabanes.
Historically, the OWC method has been implemented in `Matlab <https://github.com/ArgoDMQC/matlab_owc>`_ . More recently a `python version has been developed <https://github.com/euroargodev/argodmqc_owc>`_.

Preprocessing data
^^^^^^^^^^^^^^^^^^

At this point, both OWC software take as input a pre-processed version of the Argo float data to evaluate/calibrate.

**argopy** is able to perform this preprocessing and to create a *float source* data to be used by OWC software. This is made by :meth:`Dataset.argo.create_float_source`.

First, you would need to fetch the Argo float data you want to calibrate, in ``expert`` mode:

.. ipython:: python
    :okwarning:

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

.. _qc_traj:

Trajectories
------------

Topography
^^^^^^^^^^
.. currentmodule:: argopy

For some QC of trajectories, it can be useful to easily get access to the topography. This can be done with the **argopy** utility :class:`TopoFetcher`:

.. code-block:: python
    
    from argopy import TopoFetcher
    box = [-65, -55, 10, 20]
    ds = TopoFetcher(box, cache=True).to_xarray()

.. image:: ../../_static/topography_sample.png


Combined with the fetcher property ``domain``, it now becomes easy to superimpose float trajectory with topography:

.. code-block:: python

    fetcher = ArgoDataFetcher().float(2901623)
    ds = TopoFetcher(fetcher.domain[0:4], cache=True).to_xarray()

.. code-block:: python

    fig, ax = loader.plot('trajectory', figsize=(10, 10))
    ds['elevation'].plot.contourf(levels=np.arange(-6000,0,100), ax=ax, add_colorbar=False)

.. image:: ../../_static/trajectory_topography_sample.png


.. note::
    The :class:`TopoFetcher` can return a lower resolution topography with the ``stride`` option. See the :class:`argopy.TopoFetcher` full documentation for all the details.


Altimetry
---------
.. currentmodule:: argopy

Satellite altimeter measurements can be used to check the quality of the Argo profiling floats time series. The method compares collocated sea level anomalies from altimeter measurements and dynamic height anomalies calculated from Argo temperature and salinity profiles for each Argo float time series [Guinehut2008]_. This method is performed routinely by CLS and results are made available online.


**argopy** provides a simple access to this QC analysis with an option to the data and index fetchers :meth:`DataFetcher.plot` methods that will insert the CLS Satellite Altimeter report figure on a notebook cell.

.. code-block:: python

    fetcher = ArgoDataFetcher().float(6902745)
    fetcher.plot('qc_altimetry', embed='list')

.. image:: https://data-argo.ifremer.fr/etc/argo-ast9-item13-AltimeterComparison/figures/6902745.png

See all details about this method here: :meth:`argopy.plot.open_sat_altim_report`


.. rubric:: References

.. [Guinehut2008] Guinehut, S., Coatanoan, C., Dhomps, A., Le Traon, P., & Larnicol, G. (2009). On the Use of Satellite Altimeter Data in Argo Quality Control, Journal of Atmospheric and Oceanic Technology, 26(2), 395-402. `10.1175/2008JTECHO648.1 <https://doi.org/10.1175/2008JTECHO648.1>`_
