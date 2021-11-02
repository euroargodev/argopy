---
name: Prepare next release
about: 'release'
title: 'Prepare next release'
labels: 'next-release'
assignees: ''
---

# Prepare release

- [ ] Make sure that all [CI tests are passed with *free* environments](https://github.com/euroargodev/argopy/actions/workflows/pythonFREEtests.yml?query=event%3Apull_request)
- [ ] Update ``./requirements.txt`` and ``./docs/requirements.txt`` with CI free environments dependencies versions 
- [ ] Update ``./ci/requirements/py*-dev.yml`` with last free environments dependencies versions
- [ ] Make sure that all [CI tests are passed with *dev* environments](https://github.com/euroargodev/argopy/actions/workflows/pythontests.yml?query=event%3Apull_request)
- [ ] Increase release version in ``./setup.py`` file
- [ ] Update date and release version in ``./docs/whats-new.rst``
- [ ] Merge this PR to master

# Publish release
- [ ] Make sure that all [CI tests are passed](https://github.com/euroargodev/argopy/actions)
- [ ] On the master branch, commit the release in git:
      ```git commit -a -m 'Release v0.X.Y'```
- [ ] Tag the release:
      ```git tag -a v0.X.Y -m 'v0.X.Y'```
- [ ] Push it online:
       ```git push origin v0.X.Y```
- [ ] Issue the release on GitHub. Click on "Draft a new release" at
     https://github.com/euroargodev/argopy/releases. Type in the version number, but
     don't bother to describe it -- we maintain that on the docs instead.
      
     This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on Pypi.