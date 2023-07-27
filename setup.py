# -*coding: UTF-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="argopy",
    version="0.1.14rc2",
    author="argopy Developers",
    author_email="gmaze@ifremer.fr",
    description="A python library for Argo data beginners and experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euroargodev/argopy",
    packages=setuptools.find_packages(),
    package_dir={"argopy": "argopy"},
    package_data={"argopy": ["static/assets/*", "static/css/*"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Development Status :: 3 - Alpha",
    ],
    entry_points={
        "xarray.backends": ["argo=argopy.xarray:ArgoEngine"],
    },
)
