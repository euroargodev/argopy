---
title: 'argopy: A Python package for ocean Argo data analysis'
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
date: 17 April 2020
bibliography: paper.bib

---

# Summary

The ocean is a key component of the Earth climate system. It thus needs a continuous real-time monitoring to help scientists 
better understand its dynamic and predict its evolution. All around the world, oceanographers have manage to join their
efforts and set up a [Global Ocean Observing System](https://www.goosocean.org/) among which *Argo* is a key component. 

Argo is a real-time global ocean *in situ* observing system. It is a global network of nearly 4000 autonomous probes measuring 
pressure, temperature and salinity from the surface to 2000m depth every 10 days. The localisation of these probes is 
nearly random between the $60^o$ parallels ([see live coverage here](www.jcommops.org/ftp/Argo/Maps/countries.png)).
All probes data are collected by satellite in real-time, processed by several data centers and finally merged in a single
dataset (collecting more than 2 millions of vertical profiles data) made freely available to anyone through 
a [ftp server](ftp://ftp.ifremer.fr/ifremer/argo) or [monthly zip snapshots](http://dx.doi.org/10.17882/42182).

The Argo community focuses on delivering a curated dataset of the best scientific quality, and never provided 
its user base with a Python software to easily access and manipulate Argo measurements: the **argopy** software aims 
to fill this gap. The **argopy** software can be used to fetch, manipulate and analyse Argo floats measurements. 
It is dedicated to scientists without knowledge of the Argo data management system and can still accomodate experts 
requirements.


# Why **argopy** ?

For non-experts of the Argo dataset, it is rather complicated to get access to Argo measurements. Even though data are
made freely available on the web, the Argo dataset is made of: thousands of files organised using jargon, uses 
thousands of different variables, tens of reference tables and has an exhaustive [user manual](http://dx.doi.org/10.13155/29825) more than 100 pages long.

This complexity arises from the facts that Argo operates many different models of floats and sensors, quality control of *in situ* measurements from autonomous platforms 
requires a lot of complementary information (meta-data), and the Argo data management workflow is distributed 
between more than 10 Data Assembly Centers all around the world.

The counter part to its tremendous success in data management and in developping good practices and well calibrated 
procedures ([see all the Argo Data Management Team documentation here](http://www.argodatamgt.org/Documentation)) is a 
very complex Argo dataset: with thousands of different variables, tens of reference tables and 
a [user manual](http://dx.doi.org/10.13155/29825) more than 100 pages long: the **argopy** software aims to help users
navigate this complex realm.

Moreover, the Argo community, focusing on delivering a curated dataset of the best scientific quality, never provided 
its user base with a Python software to easily access and manipulate Argo measurements: the **argopy** software aims to fill this gap.


# More on Argo

The Argo international observation array was initiated in 1999 and soon revolutionized our 
perspective on the large scale structure and variability of the ocean by providing seasonally and regionally unbiased 
in situ temperature/salinity measurements of the ocean interior, key information that satellites can't provide [@riser-2016]. 
The Argo array reached its full global coverage (of 1 profile per month and per 3x3 degree horizontal area) in 2007, and 
pursues its evolution to fullfill new scientific requirements [@roemmich-2019].

\autoref{argo:fig:argo:A} shows the current coverage of the network. It now extents to higher latitudes than the 
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


# Acknowledgements

This software project was created with support from the EARISE project, a European Union’s Horizon 2020 research and 
innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other 
world-class research infrastructures.