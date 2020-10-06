.. _why:

Why argopy ?
============

Surprisingly, the Argo community never provided its user base with a Python software to easily access and manipulate Argo measurements:
**argopy** aims to fill this gap.

Despite, or because, its tremendous success in data management and in developping good practices and well calibrated procedures [ADMT]_, the Argo dataset is very complex: with thousands of different variables, tens of reference tables and a `user manual <http://dx.doi.org/10.13155/29825>`_ more than 100 pages long:
**argopy** aims to help you navigate this complex realm.

For non-experts of the Argo dataset, it has become rather complicated to get access to Argo measurements.
This is mainly due to:

* Argo measurements coming from many different models of floats or sensors,
* quality control of *in situ* measurements of autonomous platforms being really a matter of ocean and data experts,
* the Argo data management workflow being distributed between more than 10 Data Assembly Centers all around the world.

Less data wrangling, more scientific analysis
---------------------------------------------

In order to ease Argo data analysis for the vast majority of **standard** users, we implemented in **argopy** different levels of verbosity and data processing to hide or simply remove variables only meaningful to **experts**.
Let **argopy** manage data wrangling, and focus on your scientific analysis.

If you don't know in which category you would place yourself, try to answer the following questions:

* [ ] what is a WMO number ?
* [ ] what is the difference between Delayed and Real Time data mode ?
* [ ] what is an adjusted parameter ?
* [ ] what a QC flag of 3 means ?

If you don't answer to more than 1 question: you probably will feel more confortable with the *standard* user mode.

By default, all **argopy** data fetchers are set to work with a **standard** user mode, the other possible mode is **expert**.

In *standard* mode, fetched data are automatically filtered to account for their quality (only good are retained) and level of processing by the data centers (wether they looked at the data briefly or not).

Selecting user mode is further explained in the dedicated documentation section: :ref:`user-mode`.

.. [ADMT] See all the ADMT documentation here: http://www.argodatamgt.org/Documentation