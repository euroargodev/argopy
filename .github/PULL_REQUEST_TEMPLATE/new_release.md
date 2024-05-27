---
name: Prepare for next release
about: 'release'
title: 'Prepare for next release'
labels: 'release'
assignees: ''
---

# Setup

- [ ] Create a new branch for this release: ``git checkout -b releasev0.X.Y``
- [ ] Increase release version in ``./setup.py``
- [ ] Update release version in ``./docs/whats-new.rst``
- [ ] Create a PR to prepare it, name it with one of the [Nature emoji](https://www.webfx.com/tools/emoji-cheat-sheet/#tabs-3) and make sure it was [never used before](https://github.com/euroargodev/argopy/pulls?q=is%3Apr+label%3Arelease+) 

# Prepare code for release

## Code clean-up
- [ ] Run [codespell](https://github.com/codespell-project/codespell) from repo root and fix errors: ``codespell -q 2``
- [ ] Run [flake8](https://github.com/PyCQA/flake8) from repo root and fix errors

## Software distribution readiness
- [ ] Manually trigger [upstream CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests-upstream.yml) for the release branch and ensure they are passed
- [ ] Update pinned dependencies versions in ``./ci/requirements/py*-*-pinned.yml`` environment files using [upstream CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests-upstream.yml) information
- [ ] If CI tests with the oldest dependencies versions are not passed, upgrade these versions in ``./ci/requirements/py*-*-min.yml`` files up to the point where CI tests are passed
- [ ] Possibly update ``./requirements.txt`` and ``./docs/requirements.txt`` if the oldest dependencies versions were upgraded in the previous step
- [ ] Make sure that all CI tests are passed
- [ ] [Activate](https://readthedocs.org/projects/argopy/versions/) and make sure the documentation for the release branch is [built on RTD](https://readthedocs.org/projects/argopy/builds/)

## Preparation conclusion
- [ ] Merge this PR to master
- [ ] Update release date in ``./docs/whats-new.rst``
- [ ] Make sure all CI tests are passed and RTD doc is built on the master branch

# Publish the release

- [ ] ["Draft a new release"](https://github.com/euroargodev/argopy/releases/new) on GitHub.
Choose a release tag v0.X.Y, fill in the release title and click on the `Auto-generate release notes` button.  
This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on [Pypi](https://pypi.org/project/argopy/#history).
- [ ] Last check the ``./setup.py`` file version of the release and that the [documentation is ready](https://readthedocs.org/projects/argopy/builds/)
- [ ] Publish !
- [ ] Checkout on [Pypi](https://pypi.org/project/argopy/#history) and [Conda](https://github.com/conda-forge/argopy-feedstock/pulls) that the new release is distributed.

[![Publish on pypi](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/euroargodev/argopy/actions/workflows/pythonpublish.yml)

# CI tests / RTD build results
[![CI tests](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg?branch=releasev0.X.Y)](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml) 
[![CI tests Upstream](https://github.com/euroargodev/argopy/actions/workflows/pytests-upstream.yml/badge.svg?branch=releasev0.X.Y)](https://github.com/euroargodev/argopy/actions/workflows/pytests-upstream.yml)
[![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=releasev0.X.Y)](https://argopy.readthedocs.io/en/releasev0.X.Y)
