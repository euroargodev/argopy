.. _what_is_argo:

What is Argo ?
==============

**Argo is a real-time global ocean in situ observing system**.

The ocean is a key component of the Earth climate system. It thus needs a continuous real-time monitoring to help scientists
better understand its dynamic and predict its evolution. All around the world, oceanographers have managed to join their
efforts and set up a `Global Ocean Observing System <https://www.goosocean.org>`_ among which *Argo* is a key component.

*Argo* is a global network of nearly 4000 autonomous probes measuring
pressure, temperature and salinity from the surface to 2000m depth every 10 days. The localisation of these probes is
nearly random between the 60th parallels (`see live coverage here <https://dataselection.euro-argo.eu/>`_).
All probes data are collected by satellite in real-time, processed by several data centers and finally merged in a single
dataset (collecting more than 2 millions of vertical profiles data) made freely available to anyone through
a `ftp server <ftp://ftp.ifremer.fr/ifremer/argo>`_ or `monthly zip snapshots <http://dx.doi.org/10.17882/42182>`_.

The Argo international observation array was initiated in 1999 and soon revolutionized our
perspective on the large scale structure and variability of the ocean by providing seasonally and regionally unbiased
in situ temperature/salinity measurements of the ocean interior, key information that satellites can't provide
(`Riser et al, 2016 <http://dx.doi.org/10.1038/nclimate2872>`_).

The Argo array reached its full global coverage (of 1 profile per month and per 3x3 degree horizontal area) in 2007, and
continuously pursues its evolution to fulfill new scientific requirements (`Roemmich et al, 2019
<https://www.frontiersin.org/article/10.3389/fmars.2019.00439>`_). It now extents to higher latitudes and some of the
floats are able to profile down to 4000m and 6000m. New floats are also equipped with biogeochemical sensors, measuring
oxygen and chlorophyll for instance. Argo is thus providing a deluge of in situ data: more than 400 profiles per day.

Each Argo probe is an autonomous, free drifting, profiling float, i.e. a probe that can't control its trajectory but
is able to control its buoyancy and thus to move up and down the water column as it wishes. Argo floats continuously
operate the same program, or cycle, illustrated in the figure below. After 9 to 10 days of free drift at a parking
depth of about 1000m, a typical Argo float dives down to 2000m and then shoals back to the surface while measuring pressure,
temperature and salinity. Once it reaches the surface, the float sends by satellite its measurements to a data center
where they are processed in real time and made freely available on the web in less than 24h00.


.. figure:: _static/argofloats_cycle.png
    :alt: Argo float cycle

    Typical 10 days program, cycle, of an Argo float


.. _why:

So why argopy ?
===============

Surprisingly, the Argo community never provided its user base with a Python software to easily access and manipulate Argo measurements:
**argopy** aims to fill this gap.

Despite, or because, its tremendous success in data management and in developing good practices and well calibrated procedures [ADMT]_, the Argo dataset is very complex: with thousands of different variables, tens of reference tables and a `user manual <http://dx.doi.org/10.13155/29825>`_ more than 100 pages long:
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

If you don't answer to more than 1 question: you probably will feel more comfortable with the *standard* or *research* user modes.

By default, all **argopy** data fetchers are set to work with a **standard** user mode, the other possible modes are **research** and **expert**.

Each user modes and how to select it are further explained in the dedicated documentation section: :ref:`user-mode`.

.. [ADMT] See all the ADMT documentation here: http://www.argodatamgt.org/Documentation