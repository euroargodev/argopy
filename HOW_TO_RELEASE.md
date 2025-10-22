# Procedure to release a new version of Argopy

Argopy follows [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning) with a Major.Minor.Patch / X.Y.Z pattern. 

We try to implement the following policy:
- Increase *Patch* for bug fixes and existing feature improvements (deprecation cycle not affected).
- Increase *Minor* to highlight one important new feature, a batch of feature improvements, enforce deprecation policy.
- Increase *Major* for significant refactoring, change in API facade or to highlight software/team milestone.

Current versions of Argopy distributed with pypi and conda are:

![argopy-pypi](https://img.shields.io/pypi/v/argopy) ![argopy-conda](https://img.shields.io/conda/vn/conda-forge/argopy?logo=anaconda)

# How to release a Patch version

## Setup

Don't change X Major and Y Minor, increase Z Patch only.

- [ ] Create a new branch for this release: ``git checkout -b releasevX.Y.Z``
- [ ] Update release version in ``./docs/whats-new.rst``
- [ ] Increase release version in ``./setup.py``
- [ ] Create a PR to prepare it, copy/paste this section to the PR description.
- [ ] [Activate RTD build for this branch](https://app.readthedocs.org/dashboard/argopy/version/create/)

## Prepare code for release

### Deprecation policy
Does not apply for Patch release. Only consider deprecation policy for Major or Minor releases.

### Update static content
- [ ] Update [static asset files](https://github.com/euroargodev/argopy/tree/master/argopy/static/assets) using the CLI [update_json_assets](https://github.com/euroargodev/argopy/tree/master/argopy/cli/update_json_assets) command.
- [ ] Update the [cheatsheet PDF](https://github.com/euroargodev/argopy/blob/master/docs/_static/argopy-cheatsheet.pdf) with all new release features, if any.

### Code clean-up and update
- [ ] Run [codespell](https://github.com/codespell-project/codespell) from repo root and fix errors: ``codespell -q 2``
- [ ] Run [flake8](https://github.com/PyCQA/flake8) from repo root and fix errors

### Software distribution readiness
- [ ] Possibly update ``./requirements.txt`` and ``./docs/requirements.txt`` if this Patch release requires a change in dependencies version to fix a bug.
- [ ] Make sure that all CI tests are passed: [![CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg?branch=releasevX.Y.Z)](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml)
- [ ] Make sure the documentation for this release branch is [built on RTD](https://app.readthedocs.org/projects/argopy/builds/): [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=releasevX.Y.Z)](https://argopy.readthedocs.io/en/releasevX.Y.Z)


### Preparation conclusion
- [ ] Merge this PR to master
- [ ] Update release date in ``./docs/whats-new.rst``
- [ ] Verify that all CI tests are passed [![CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg?branch=master)](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml) 
- [ ] Verify that RTD doc is built on the master branch [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest)

## Publish the release
- [ ] Last check the ``./setup.py`` file version of the release and that the [documentation is ready](https://readthedocs.org/projects/argopy/builds/)
- [ ] ["Create a new release"](https://github.com/euroargodev/argopy/releases/new) on GitHub.
Choose a release tag vX.Y.Z, fill in the release title and click on the `Auto-generate release notes` button. Once ready, publish the release. This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on [Pypi](https://pypi.org/project/argopy/#history).
- [ ] Checkout on [Pypi](https://pypi.org/project/argopy/#history) and [Conda](https://github.com/conda-forge/argopy-feedstock/pulls) that the new release is distributed.

[![Publish on pypi](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml)

# How to release Major or Minor versions

**For a Major**: increase X Major, reset Y Minor and Z Patch to 0 (eg: v2.0.0).

**For a Minor**: don't change X Major, increase Y Minor only, reset Z Patch to 0 (eg: v1.3.0).

## Setup

- [ ] Create a new branch for this release: ``git checkout -b releasevX.Y.Z``
- [ ] Update release version in ``./docs/whats-new.rst``
- [ ] Increase release version in ``./setup.py``
- [ ] Create a PR to prepare it, copy/paste this section to the PR description.
- [ ] [Activate RTD build for this branch](https://app.readthedocs.org/dashboard/argopy/version/create/)

## Prepare code for release

### Deprecation policy
- [ ] Check the code for the ``deprecated`` decorator and enforce the deprecation policy:
  - [ ] If code is marked as deprecated since version = vX.Y.Z : do nothing (first version with deprecation warning)
  - [ ] If code is marked as deprecated since version = vX.(Y-1).Z : do nothing (2nd and last version with deprecation warning)
  - [ ] If code is marked as deprecated since version = vX.(Y-2).Z : delete code (code will raise an error)
- [ ] Update the documentation file ``whats-new.rst`` section `Internals` of this release with the list of class/methods that have been deleted.

### Update static content

#### CI tests data
- [ ] Update CI tests data used by mocked ftp and http servers. Use the CLI [citests_httpdata_manager](https://github.com/euroargodev/argopy/blob/master/cli/citests_httpdata_manager):
  ```bash
  cd cli
  ./citests_httpdata_manager -a clear --force --refresh
  ./citests_httpdata_manager -a download
  ./citests_httpdata_manager -a check
  ```
  
#### Library static assets
- [ ] Update [static asset files](https://github.com/euroargodev/argopy/tree/master/argopy/static/assets) using the CLI [update_json_assets](https://github.com/euroargodev/argopy/tree/master/argopy/cli/update_json_assets) command.
- [ ] Update [cheatsheet PDF](https://github.com/euroargodev/argopy/blob/master/docs/_static/argopy-cheatsheet.pdf) with all new release features.

### Code clean-up and update
- [ ] Run [codespell](https://github.com/codespell-project/codespell) from repo root and fix errors: ``codespell -q 2``
- [ ] Run [flake8](https://github.com/PyCQA/flake8) from repo root and fix errors

### Software distribution readiness
- [ ] Possibly update ``./requirements.txt`` and ``./docs/requirements.txt`` if the dependencies versions were upgraded for a bug fix or new feature.
- [ ] Make sure that all CI tests are passed: [![CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg?branch=releasevX.Y.Z)](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml)
- [ ] Make sure the documentation for this release branch is [built on RTD](https://app.readthedocs.org/projects/argopy/builds/): [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=releasevX.Y.Z)](https://argopy.readthedocs.io/en/releasevX.Y.Z)


### Preparation conclusion
- [ ] Merge this PR to master
- [ ] Update release date in ``./docs/whats-new.rst``
- [ ] Verify that all CI tests are passed [![CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg?branch=master)](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml) 
- [ ] Verify that RTD doc is built on the master branch [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest)

## Publish the release

- [ ] Last check the ``./setup.py`` file version of the release and that the [documentation is ready](https://readthedocs.org/projects/argopy/builds/)
- [ ] ["Create a new release"](https://github.com/euroargodev/argopy/releases/new) on GitHub.
Choose a release tag vX.Y.Z, fill in the release title and click on the `Auto-generate release notes` button. Once ready, publish the release. This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on [Pypi](https://pypi.org/project/argopy/#history).
- [ ] Checkout on [Pypi](https://pypi.org/project/argopy/#history) and [Conda](https://github.com/conda-forge/argopy-feedstock/pulls) that the new release is distributed (this can take 2/3 days for conda).

[![Publish on pypi](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml)