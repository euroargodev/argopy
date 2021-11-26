1. [ ] Run codespell ``codespell -q 3``

2. [ ] Make sure that all CI tests are passed with **free* environments

3. [ ] Update ``./requirements.txt`` and ``./docs/requirements.txt`` with CI free environments dependencies versions

4. [ ] Update ``./ci/requirements/py*-dev.yml`` with last free environments dependencies versions

5. [ ] Make sure that all CI tests are passed with **dev* environments

6. [ ] Increase release version in ``./setup.py`` file

7. [ ] Update date and release version in ``./docs/whats-new.rst``

8. [ ] On the master branch, commit the release in git:

      ```git commit -a -m 'Release v0.X.Y'```

9. [ ] Tag the release:

      ```git tag -a v0.X.Y -m 'v0.X.Y'```

10. [ ] Push it online:

       ```git push origin v0.X.Y```

11. [ ] Issue the release on GitHub. Click on "Draft a new release" at
     https://github.com/euroargodev/argopy/releases. Type in the version number, but
     don't bother to describe it -- we maintain that on the docs instead.
    
     This will trigger the publish Github action that will push the release on Pypi.