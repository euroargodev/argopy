# -*coding: UTF-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="argopy",
    version="1.4.0",
    author="argopy Developers",
    author_email="gmaze@ifremer.fr",
    description="A python library for Argo data beginners and experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euroargodev/argopy",
    packages=setuptools.find_packages(),
    package_dir={"argopy": "argopy"},
    package_data={"argopy": ["static/assets/*", "static/css/*"]},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Development Status :: 5 - Production/Stable",
    ],
    entry_points={
        "xarray.backends": ["argo=argopy.xarray:ArgoEngine"],
    },
)
