from setuptools import find_packages, setup
from pathlib import Path


# Package meta-data.
NAME = "zms2"
DESCRIPTION = "Image analysis pipeline for MS2 reporters in large, dense tissues like zebrafish embryos."
URL = "https://github.com/bschloma/zms2"
EMAIL = "bschloma@berkeley.edu"
AUTHOR = "Brandon Schlomann"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.1"

CUPY_VERSION = "11.5.0"

long_description = (Path(__file__).parent / "README.md").read_text()

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy>=1.20",
    "scipy>=1.8.0",
    "scikit-image",
    "tqdm",
    "dask",
    "zarr",
    "scikit-learn",
    "pandas",
    #"cupy",
    #"cucim",
    #"napari",
    #"tensorflow>=2.11.0",
]

# What packages are optional?
EXTRAS = {}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="BSD 3-Clause",
    license_file="LICENSE.txt",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
