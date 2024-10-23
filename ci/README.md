
# How to manage CI/dev environments ?

Use the ``envs_manager`` bash shell script located in this folder.

## Environment files update procedure

To update the suite of environments (for a new release for instance), the procedure could be:

- Build the ``free`` versions environment (e.g. CLI: ``envs_manager -i free`` will install all envs for all Python versions)
- For python versions in py=[3.9, 3.10]:
  - For **all** packages:
    - Activate the ``free`` environment: 
    - Execute the CLI ``show_versions --conda --free`` and update ``py<py>-all-free.yml`` file accordingly
    - Execute the CLI ``show_versions --conda`` and update ``py<py>-all-pinned.yml`` file accordingly
  - For **core** packages:
    - Activate the ``core`` environment: 
    - Execute the CLI ``show_versions --conda --free --core`` and update ``py<py>-core-free.yml`` file accordingly
    - Execute the CLI ``show_versions --conda --core`` and update ``py<py>-core-pinned.yml`` file accordingly

```bash
./envs_manager -i py310-core-free
```