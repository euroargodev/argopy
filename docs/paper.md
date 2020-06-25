---
title: 'argopy: A Python library for Argo ocean data analysis'
tags:
  - Python
  - ocean
  - oceanography
  - observation
authors:
  - name: Guillaume Maze
    orcid: 0000-0001-7231-2095
    affiliation: 1
  - name: Kevin Balem
    orcid: 0000-0002-4956-8698
    affiliation: 1
affiliations:
 - name: Ifremer, University of Brest, CNRS, IRD, Laboratoire d'Océanographie Physique et Spatiale, IUEM, 29280, Plouzané, France
   index: 1
date: 24 June 2020
bibliography: paper.bib

---

# Summary

Argo is a real-time global ocean *in situ* observing system. It provides highly accurate thousands of ocean measurements 
every day. The Argo dataset has now accumulated more than X billions of measurements and accessing it for scientific 
analysis remains a challenge.

The Argo expert community focuses on delivering a curated dataset of the best scientific quality, and never provided 
its user base with a Python software to easily access and manipulate Argo measurements: the **argopy** software aims 
to fill this gap. The **argopy** software can be used to easily fetch and manipulate Argo floats measurements. 
It is dedicated to scientists without knowledge of the Argo data management system and can still accommodate experts 
requirements.

# Introduction

The ocean is a key component of the Earth climate system. It thus needs a continuous real-time monitoring to help scientists 
better understand its dynamic and predict its evolution. All around the world, oceanographers have manage to join their
efforts and set up a [Global Ocean Observing System](https://www.goosocean.org/) among which *Argo* is a key component. 

Argo is a global network of nearly 4000 autonomous probes measuring pressure, temperature and salinity from the surface 
to 2000m depth every 10 days. The localisation of these probes is nearly random between the $60^o$ parallels ([see live 
coverage here](http://map.argo-france.fr)). All probes data are collected by satellite in real-time, processed by several 
data centers and finally merged in a single dataset (collecting more than 2.3 millions of vertical profiles data as of 
June 2020) made freely available to anyone through a [ftp server](ftp://ftp.ifremer.fr/ifremer/argo) or [monthly zip 
snapshots](http://dx.doi.org/10.17882/42182).

The Argo international observation array was initiated in 1999 and soon revolutionised our 
perspective on the large scale structure and variability of the ocean by providing seasonally and regionally unbiased 
in situ temperature/salinity measurements of the ocean interior, key information that satellites can't provide [@riser-2016]. 
The Argo array reached its full global coverage (of 1 profile per month and per 3x3 degree horizontal area) in 2007, and 
pursues its evolution to fulfil new scientific requirements [@roemmich-2019]. Argo data have been used in more than 4000 scientific publications.

This [online figure](http://map.argo-france.fr) shows the current coverage of the network. It now extents to higher latitudes than the 
original $\pm60^o$ and some of the floats are able to profile down to 4000m and 6000m. New floats are also equipped 
with biogeochemical sensors, measuring oxygen and chlorophyll for instance. All these evolutions of the network increase 
the total number of floats to nearly 4000. Argo is thus providing a deluge of in situ data: more than 400 profiles per day.

Each Argo probe is an autonomous, free drifting, profiling float, i.e. a probe that can't control its trajectory but 
is able to control its buoyancy and thus to move up and down the water column as it wishes. Argo floats continuously 
operate the same program, or cycle, illustrated \autoref{fig:argofloat}. After 9 to 10 days of free drift at a parking 
depth of about 1000m, a typical Argo float dives down to 2000m and then shoals back to the surface while measuring pressure, 
temperature and salinity. Once it reaches the surface, the float sends by satellite its measurements to a data center 
where they are processed in real time and made freely available on the web in less than 24h00.

![Typical 10 days program, cycle, of an Argo float.\label{fig:argofloat}](_static/argofloats_cycle.png)


# Why **argopy** ?

For non-experts of the Argo dataset, it is rather complicated to get access to Argo measurements. Even though data are
made freely available on the web, the Argo dataset is made of: thousands of files organised using jargon, uses 
thousands of different variables, tens of reference tables and has an exhaustive [user manual](http://dx.doi.org/10.13155/29825) more than 100 pages long.

This complexity arises from the facts that Argo operates many different models of floats and sensors, quality control 
of *in situ* measurements from autonomous platforms requires a lot of complementary information (meta-data), and the 
Argo data management workflow is distributed between more than 10 Data Assembly Centers all around the world. The Argo 
data management is a model for other ocean observing systems and constantly ensures the highest quality of scientific 
measurements for the community [@wong-2020].

The counter part to this tremendous success in data managemen, in developing good practices and well calibrated 
procedures ([see all the Argo Data Management Team documentation here](http://www.argodatamgt.org/Documentation)) is thus 
a very complex Argo dataset: the **argopy** software aims to help users navigate this complex realm.

Moreover, since the Argo community focuses on delivering a curated dataset for science, it does not provide open source softwares 
for its user base. Softwares exist for Argo data operators so that they can decode and quality control the data [e.g. @scoop].
But none is available for scientists who thus have to develop their own machinery to download and manipulate the data.

Python is becoming widely used by the scientific community and beyond: worldwide, Python is the most popular language that 
grew the most in the last 5 years (20%, source: http://pypl.github.io/PYPL.html). It offers a modern, powerful and open
source framework to work with. However, up to this point, no Python based software has been dedicated to the Argo dataset.  

# Key features of **argopy**

**argopy** is a python software that aims to ease Argo data access and manipulation for standard users as well as Argo 
experts. The two key features of **argopy** are thus (i) its trivial fetching API of Argo data and (ii) its 
ability to provide data formatted for both beginners and experts of Argo.

## Data fetching

**argopy** provides a trivial fetching API of Argo data through a simple call to one of the 3 different ways of 
looking at Argo data: over a space/time domain (with the *region* access point), for one or a list of specific floats (given 
their unique [WMO number](https://www.wmo.int/pages/prog/amp/mmop/wmo-number-rules.html) with the *float* access point) 
or for one or a list of float profiles (with the *profile* access point). This is as simple as:
```python
from argopy import DataFetcher as ArgoDataFetcher
fetcher = ArgoDataFetcher().region([-75, -45, 20, 30, 0, 100, '2011', '2012'])
ds = fetcher.to_xarray()
```
Here we used **argopy** to fetch data between 75/45W, 20/30N, from 0 to 100db and for the entire year 2011.
Once the user has defined what it needs (the ``fetcher`` class instance), **argopy** will fetch data online and manage 
internally all the complicated processing of formatting the web request and creating a workable in memory data 
structure (the ``to_xarray()`` call above). By default, **argopy** uses the [xarray data model](http://xarray.pydata.org).
*xarray* is an open source Python package to easily work with labelled multi-dimensional arrays.

## Data formatting

**argopy** aims to thrive in providing Argo data to non-experts. So one key feature of **argopy** is the option for selecting
a *user mode* that is either ``standard`` or ``expert``. Standard users are those who want to focus on the measurements 
for scientific analysis, those who do not know, or don't want to be bothered with all the Argo jargon and multitude of 
variables and parameters. 

For standard users (the default mode), **argopy** will internally runs a series of processes that
will curate raw data and provide a simplified and science focused dataset. For expert users, **argopy** will apply its 
data model to raw fetched data and return Argo variables like experts users are already used to.

## And more

**argopy** has features to manipulate Argo data, for instance:
- the possibility to transform data from a collection of measurements to a collection of vertical profiles, and vice-versa; 
- the possibility to interpolate irregularly sampled measurements onto standard pressure levels.
 
Another feature is the ability to cache fetched data, so that requests provide users with data much more rapidly, 
saving bandwidth and time. 

Two last important features of **argopy** to describe here are: 
- the possibility to fetch data locally, from a user copy of the entire or subset of the Argo database,
- the possibility to fetch only meta data (organised in *index* lookup tables), which allows to determine the regional Argo sampling 
for instance.
These more advance features may be more of interest for ``expert`` users, since it more knowledge of the Argo dataset.

# Acknowledgements

We acknowledge support from the [Euro-Argo ERIC community](https://www.euro-argo.eu/) during the genesis of this project.
This software was created with support from the EARISE project, a European Union’s Horizon 2020 research and 
innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other 
world-class research infrastructures.