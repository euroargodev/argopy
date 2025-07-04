Argo data python library
========================

**argopy** is a python library dedicated to :ref:`Argo <what_is_argo>` data access, manipulation and visualisation
for standard users as well as Argo experts.

|JOSS| |lifecycle| |Gitter|

|License| |Python version| |Anaconda-Server Badge|


.. admonition:: 2025 argopy training camps 🎓

    The argopy team will organise "training camps" in 2025:

    1. **in person** training: Monday Sept. 22nd afternoon, `Thalassocosmos <https://cretaquarium.gr/>`_, HCMR, Heraklion, Crete, during the `Euro-Argo Science Meeting <https://www.euro-argo.eu/News-Meetings/News/News-archives/2025/8th-Euro-Argo-Science-Meeting>`_: `registration are now open here <https://forms.ifremer.fr/euroargo/8th-euro-argo-science-meeting-registration-form/>`_.

    2. **in person** training: Wednesday June 25th 10am, room 304, `JAMSTEC Headquarters <https://www.jamstec.go.jp/e/about/access/yokosuka.html>`_ `GOORC <https://www.jamstec.go.jp/goorc/e/>`_, Yokosuka, Japan. No registration required.

    3. **in person** training: Wednesday June 18th 10am, room 425, `Tohoku University <https://www.gp.tohoku.ac.jp/pol/index-e.html>`_ `WPI-AIMEC <https://wpi-aimec.jp/en/>`_, Sendai, Japan. No registration required.

    4. **online** training, to be organised sometime this fall: `register here <https://forms.gle/d8xPbrWu7aZcvMut9>`_

    The goal of these events is to train users with all the argopy features.
    Whether you're a standard, research or expert users, argopy has features for you !


Documentation
-------------

**Getting Started**

* :doc:`install`
* :doc:`usage`
* :doc:`Gallery of examples <gallery>`
* :doc:`What is Argo ? Why argopy ? <what_why>`
* :doc:`impact`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started

    install
    usage
    Gallery of examples and tutorials <gallery>
    What is Argo ? Why argopy ? <what_why>
    impact

**User Guide**

* :doc:`user-guide/fetching-argo-data/index`
* :doc:`user-guide/working-with-argo-data/index`

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: User Guide

    Fetching Argo data <user-guide/fetching-argo-data/index>
    user-guide/working-with-argo-data/index

**Advanced Tools**

* :doc:`Argo file stores <advanced-tools/stores/index>`
* :doc:`advanced-tools/metadata/index`
* :doc:`advanced-tools/quality_control/index`
* :doc:`Improving performances  <advanced-tools/performances/index>`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Advanced Tools

    Argo file stores <advanced-tools/stores/index>
    advanced-tools/metadata/index
    advanced-tools/quality_control/index
    Improving performances <advanced-tools/performances/index>


**Help & reference**

* :doc:`Cheat sheet PDF <cheatsheet>`
* :doc:`whats-new`
* :doc:`energy`
* :doc:`contributing`
* :doc:`api`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Help & reference

    Cheat sheet PDF <cheatsheet>
    whats-new
    energy
    contributing
    api

.. |JOSS| image:: https://img.shields.io/badge/DOI-10.21105%2Fjoss.02425-brightgreen
   :target: //dx.doi.org/10.21105/joss.02425
.. |Documentation| image:: https://img.shields.io/static/v1?label=&message=Read%20the%20documentation&color=blue&logo=read-the-docs&logoColor=white
   :target: https://argopy.readthedocs.io
.. |Gitter| image:: https://badges.gitter.im/Argo-floats/argopy.svg
   :target: https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |License| image:: https://img.shields.io/badge/License-EUPL%201.2-brightgreen
    :target: https://opensource.org/license/eupl-1-2/
.. |Python version| image:: https://img.shields.io/pypi/pyversions/argopy
   :target: //pypi.org/project/argopy/
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/argopy/badges/platforms.svg
   :target: https://anaconda.org/conda-forge/argopy
.. |pypi dwn| image:: https://img.shields.io/pypi/dm/argopy?label=Pypi%20downloads
   :target: //pypi.org/project/argopy/
.. |conda dwn| image:: https://img.shields.io/conda/dn/conda-forge/argopy?label=Conda%20downloads
   :target: //anaconda.org/conda-forge/argopy
.. |image8| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |PyPI| image:: https://img.shields.io/pypi/v/argopy
   :target: //pypi.org/project/argopy/
.. |Conda| image:: https://anaconda.org/conda-forge/argopy/badges/version.svg
   :target: //anaconda.org/conda-forge/argopy
.. |tests in FREE env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-free.yml/badge.svg
.. |tests in DEV env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-dev.yml/badge.svg
.. |Documentation Status| image:: https://img.shields.io/readthedocs/argopy?logo=readthedocs
   :target: https://argopy.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/euroargodev/argopy
.. |lifecycle| image:: https://img.shields.io/badge/lifecycle-stable-green.svg
   :target: https://www.tidyverse.org/lifecycle/#stable
.. |Erddap status| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_erddap.json
.. |Argovis status| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_argovis.json
.. |Profile count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json
.. |Statuspage| image:: https://img.shields.io/static/v1?label=&message=Check%20all%20Argo%20monitors&color=blue&logo=statuspage&logoColor=white
   :target: https://argopy.statuspage.io
.. |image20| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |image21| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |badge| image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge
   :target: https://mybinder.org/v2/gh/euroargodev/binder-sandbox/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Feuroargodev%252Fargopy%26urlpath%3Dlab%252Ftree%252Fargopy%252Fdocs%252Ftryit.ipynb%26branch%3Dmaster
.. |index_traj| image:: https://github.com/euroargodev/argopy/raw/master/docs/_static/trajectory_sample.png
