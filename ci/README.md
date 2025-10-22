# Continuous Integration Guidelines

## How to manage CI/dev environments ?

Use the ``envs_manager`` bash shell script located in this folder.

```bash
>>> envs_manager -h
Manage argopy related Conda environments

Syntax: manage_ci_envs [-hl] [-d] [-rik]
options:
h     Print this Help
l     List all available environments
d     Dry run, just list what the script would do

r     Remove an environment
i     Install an environment (start by removing it if it's already installed)
k     Install an environment as a Jupyter kernel
```

### Environment files update procedure

To update the suite of dev/test environments, the procedure is:

- Build ``free`` versions environment (e.g. CLI: ``envs_manager -i free`` will install all envs for all Python versions)
- For python versions in py=[3.11, 3.12]:
  - For **all** packages:
    - Activate the ``free`` environment: 
    - Execute the CLI ``show_versions --conda --free`` and update ``py<py>-all-free.yml`` file accordingly
    - Execute the CLI ``show_versions --conda`` and update ``py<py>-all-pinned.yml`` file accordingly
  - For **core** packages:
    - Activate the ``core`` environment: 
    - Execute the CLI ``show_versions --conda --free --core`` and update ``py<py>-core-free.yml`` file accordingly
    - Execute the CLI ``show_versions --conda --core`` and update ``py<py>-core-pinned.yml`` file accordingly

## Argopy dependencies support policy

This is an attempt at defining the Argopy policy with regard to dependency version supports. 
This section should basically describe in which python environment Argopy is expected to run smoothly.

Over the last 5 years, Argopy has been maintained to make each new release compatible with the last versions of all its core and extra dependencies. 
This requires a lot of work, which makes very visible the diverging time scales between fast and slow release cycle of libraries like xarray (once a month or more), numpy/pandas (one minor every 6 months) and Argopy. An obvious consequence of very different dev. community sizes.

But the Argopy team want to **re-allocate resources** (i.e. time for coding) assigned to dependency support to the development of new features.

Therefore, starting with Argopy version 1.3.1 released on October 2025, we will update requirements and CI tests environment definitions **once a year**, independently of our release cycle.

In practice for users, this means that Argopy may or may not work with dependencies released after this once-a-year upgrade. 
In practice for developers, this means that ``requirements.py`` and ``ci/py<py>-<core/all>-free.yml`` environment files will have upper bounds on each dependency that shall be updated once year.

## Argopy dependencies

We distinguish 2 categories of dependencies:

- `core` dependencies are required by Argopy to load and run primary facade APIs, i.e. DataFetcher, ArgoIndex, ArgoFloat.
- `extra` dependencies are required to improve performances or for all other level APIs, such as plotting methods or related content like carbonates neural network predictors.
