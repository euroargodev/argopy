.. _energy:

Carbon emissions
================

|energyused_CItests| |energyused_CItests_upstream|


Why
---
The **argopy** team is concerned about the environmental impact of your favorite software development activities.

Starting June 1st 2024, we're experimenting with the `Green Metrics Tools <https://metrics.green-coding.io>`_ from `Green Coding <https://www.green-coding.io>`_ to get an estimate of the energy used and CO2eq emitted by our development activities on Github infrastructure.

This experiment shall help us understand the carbon emissions of our digital activities in order to minimize it.

Method
------
Our continuous integration pipeline works with Github Actions that operate on Microsoft Azure VMs. For each run, we use the `Eco-CI Energy Estimation tool <https://github.com/marketplace/actions/eco-ci-energy-estimation>`_  to monitor CPU usage. Based on the type of machine and their location, a model is used to predict the energy consumed and CO2eq emitted. This method is based on a `peer-reviewed research paper <https://www.green-coding.io/projects/cloud-energy>`_.


Results
-------

- `Dashboard of all argopy energy consumption and CO2eq emission <https://metrics.green-coding.io/carbondb-lists.html?project_uuid=a5c7557d-f668-482b-b740-b87d0bbf5b6d>`_

- `Energy used by CI tests running on each commit <https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=22344160>`_

- `Energy used by upstream CI tests running daily and on each commit <https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=25052179>`_


.. |energyused_CItests| image:: https://api.green-coding.io/v1/ci/badge/get?repo=euroargodev/argopy&branch=master&workflow=22344160
   :target: https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=22344160

.. |energyused_CItests_upstream| image:: https://api.green-coding.io/v1/ci/badge/get?repo=euroargodev/argopy&branch=master&workflow=25052179
   :target: https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=25052179
