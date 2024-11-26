Argo data python library
========================

**argopy** is a python library dedicated to :ref:`Argo <what_is_argo>` data access, manipulation and visualisation
for standard users as well as Argo experts.

|JOSS| |lifecycle| |Gitter|

|License| |Python version| |Anaconda-Server Badge|


.. admonition:: 🎉 argopy turns 5! 🎉

    Join us as we celebrate this milestone with exciting activities:

    - 🚀 `Coding Challenges <https://euroargodev.github.io/argopy-5years/challenges/index.html#coding-challenges>`_. Test your skills and creativity with a set of exciting Argo related challenges designed for all levels. Compete for bragging rights and prizes!
    - 🎮 `Online Game Contest <https://argopy.pythonanywhere.com/>`_. Join the community for a fun-filled competition that blends tech and play. Perfect for taking a break and get a special price if you make it to the top 3.
    - 📋 `User Survey <https://forms.gle/v8NnXkXCEYwRfePp8>`_. Share your feedback and ideas to help shape the future of argopy. Your input means the world to us.
    - 📚 `Free Training Camp <https://forms.gle/d8xPbrWu7aZcvMut9>`_. Expand your knowledge with expert-led sessions on making the most of argopy. Perfect for new and experienced users alike!

    We’d love to have you join us in celebrating this milestone. Whether you’ve been with us since day one or just started using argopy, your involvement makes a difference

    👉 `<https://euroargodev.github.io/argopy-5years>`_

    **Thank you for your support and for being an essential part of our journey. Here’s to the next five years of innovation, learning, and collaboration!**



.. admonition:: 2025 argopy training camps 🎓

    The argopy team will organise "training camps" in 2025:

    At  least one event would be in-person and another online.

    Overall, a training camp should be no more than 1 day long.

    The goal of these events is to train users with all the argopy features.
    Whether you're a standard, research or expert users, argopy has features for you !

    `You can pre-register here <https://forms.gle/d8xPbrWu7aZcvMut9>`_

.. versionadded:: v1.0.0

    The team proudly assumes that **argopy** is all grown up !

    This version comes with improved performances and support for the BGC-Argo dataset.
    But since this is a major, we also introduces breaking changes and significant internal refactoring possibly with un-expected side effects ! So don't hesitate to `report issues on the source code repository <https://github.com/euroargodev/argopy/issues>`_.


Documentation
-------------

**Getting Started**

* :doc:`install`
* :doc:`usage`
* :doc:`why`
* :doc:`what_is_argo`
* :doc:`Gallery of examples <gallery>`
* :doc:`impact`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started

    install
    usage
    why
    what_is_argo
    Gallery of examples <gallery>
    impact

**User Guide**

* :doc:`user-guide/fetching-argo-data/index`
* :doc:`user-guide/working-with-argo-data/index`
* :doc:`metadata_fetching`
* :doc:`performances`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    Fetching Argo data <user-guide/fetching-argo-data/index>
    user-guide/working-with-argo-data/index
    metadata_fetching
    performances

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
