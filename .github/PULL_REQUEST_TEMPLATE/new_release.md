---
name: Prepare for next release
about: 'release'
title: 'Prepare for next release'
labels: 'release'
assignees: ''
---

- [ ] Create a new branch for this release: ``git checkout -b releaseX.Y.Z``
- [ ] Create a PR to prepare it, name it with one of the [Nature emoji](https://www.webfx.com/tools/emoji-cheat-sheet/#tabs-3) 

# Prepare release

- [ ] Run codespell from repo root and fix errors: ``codespell -q 2``
- [ ] Make sure that all [CI tests are passed with *free* environments](https://github.com/euroargodev/argopy/actions?query=workflow%3A%22tests+in+FREE+env%22+event%3Apull_request)
- [ ] Update ``./requirements.txt`` and ``./docs/requirements.txt`` with CI free environments dependencies versions 
- [ ] Update ``./ci/requirements/py*-dev.yml`` with last free environments dependencies versions
- [ ] Make sure that all [CI tests are passed with *dev* environments](https://github.com/euroargodev/argopy/actions?query=workflow%3A%22tests+in+DEV+env%22+event%3Apull_request)
- [ ] Increase release version in ``./setup.py`` file
- [ ] Update date and release version in ``./docs/whats-new.rst``
- [ ] Merge this PR to master

# Publish release

- [ ] On the master branch, commit the release in git: ``git commit -a -m 'Release v0.X.Y'``
- [ ] Tag the release: ``git tag -a v0.X.Y -m 'v0.X.Y'``
- [ ] Push it online: ``git push origin v0.X.Y``
- [ ] Issue the release on GitHub by first ["Drafting a new release"](https://github.com/euroargodev/argopy/releases/new)
Choose the release tag v0.X.Y, fill in the release title and click on the `Auto-generate release notes` button.  
This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on [Pypi](https://pypi.org/project/argopy/#history).