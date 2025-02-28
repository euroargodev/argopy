.. currentmodule:: argopy.related

GDAC snapshot DOI
=================

The Argo Data Management Team maintain an exemplary DOI system to support science reproducibility and `FAIRness <https://en.wikipedia.org/wiki/FAIR_data>`_. On a monthly basis, the ADMT zip the entire Argo GDAC, archive it and assign it a specific DOI.

There is a major `DOI 10.17882/42182 <https://doi.org/10.17882/42182>`_ and each monthly snapshot has a minor DOI, with
a hashtag.

**argopy** provides the :class:`ArgoDOI` class to help you access, search and retrieve a DOI for Argo.


DOI discovery
-------------

If you don't know where to start, just load the major Argo DOI record, it will point toward the latest snapshots and list the most recent associated files:

.. ipython:: python
    :okwarning:

    from argopy import ArgoDOI

    doi = ArgoDOI()
    doi

A typical use case will be for users to access the data on a specific date and then to conduct their analysis. At the time of
writing a report or research publication, it is not trivial to get the most appropriate DOI for the dataset analysed. In
this case, the :class:`ArgoDOI` will get handy with its search method that will return the closest Argo DOI to a given date:

.. ipython:: python
    :okwarning:

    doi.search('2020-02')

You can also specify the BGC network in order to select DOI with synthetic profiles only:

.. ipython:: python
    :okwarning:

    doi.search('2020-02', network='BGC')

DOI data
--------

Once you have identified a specific hashtag for you snapshot of interest, you can point directly to it:

.. ipython:: python
    :okwarning:

    doi = ArgoDOI('109847')
    doi

The later ``doi`` object holds attributes such as ``dx`` and ``file``.

You can also trigger the tar.gz archive download with the :class:`ArgoDOI.download` method.
