- [ ] Create a new branch for this release: ``git checkout -b releaseX.Y.Z``
- [ ] Create a PR to prepare it, name it with one of the [Nature emoji](https://www.webfx.com/tools/emoji-cheat-sheet/#tabs-3) and make sure it was [never used before](https://github.com/euroargodev/argopy/pulls?q=is%3Apr+label%3Arelease+) 

# Prepare release

- [ ] Run [codespell](https://github.com/codespell-project/codespell) from repo root and fix errors: ``codespell -q 2``
- [ ] Run [flake8](https://github.com/PyCQA/flake8) from repo root and fix errors
- [ ] Increase release version in ``./setup.py``
- [ ] Update date and release version in ``./docs/whats-new.rst``
- [ ] Make sure that all CI tests are passed
- [ ] Update ``./requirements.txt`` and ``./docs/requirements.txt`` with CI free environments dependencies versions 
- [ ] Update ``./ci/requirements/py*-all/core-pinned.yml`` with last free environments dependencies versions
- [ ] Make sure that all CI tests are passed
- [ ] Make sure documentation is built on [RTD](https://readthedocs.org/projects/argopy/builds/)
- [ ] Merge this PR to master
- [ ] Make sure all CI tests are passed on the master branch

# Publish release

- [ ] On the master branch, commit the release in git: ``git commit -a -m 'Release v0.X.Y'``
- [ ] Tag the release: ``git tag -a v0.X.Y -m 'v0.X.Y'``
- [ ] Push it online: ``git push origin v0.X.Y``
- [ ] Issue the release on GitHub by first ["Drafting a new release"](https://github.com/euroargodev/argopy/releases/new)
Choose the release tag v0.X.Y, fill in the release title and click on the `Auto-generate release notes` button.  
This will trigger the [publish Github action](https://github.com/euroargodev/argopy/blob/master/.github/workflows/pythonpublish.yml) that will push the release on [Pypi](https://pypi.org/project/argopy/#history).
