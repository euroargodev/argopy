import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='argopy',
    version='0.1.1',
    author="argopy Developers",
    author_email="gmaze@ifremer.fr",
    description="A python library for Argo data beginners and experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euroargodev/argopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    install_requires=["xarray>=0.14", "numpy>=1.16.2", "scipy>=1.2.1", "gsw>=3.3",
                        "pandas>=0.24", "dask>=2.0", "erddapy>=0.5"]
)