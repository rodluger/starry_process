"""Install script for `starry_process`."""
import os
from setuptools import find_packages, setup

setup(
    name="starry_process",
    author="Rodrigo Luger",
    author_email="rodluger@gmail.com",
    url="https://github.com/rodluger/starry_process",
    description="interpretable gaussian processes for stellar light curves",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    use_scm_version={
        "write_to": os.path.join(
            "starry_process", "starry_process_version.py"
        ),
        "write_to_template": '__version__ = "{version}"\n',
    },
    install_requires=[
        "setuptools_scm",
        "numpy>=1.19.2",
        "scipy>=1.5.0",
        "Theano-PyMC",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "app": ["bokeh>=2.2.1"],
        "tests": [
            "parameterized",
            "nose",
            "pytest",
            "pytest-dependency",
            "pytest-env",
            "pymc3>=3.10.0",  # ==3.9.3",  # DEBUG: 3.10.0 --> cannot import name 'TestValueError' from 'theano.gof.utils'
            "corner",
            "pymc3-ext>=0.0.2",
            "starry@git+https://github.com/rodluger/starry@master",
            "healpy",
            "xarray==0.16.0",  # DEBUG: https://github.com/arviz-devs/arviz/issues/1387
        ],
    },
    entry_points={
        "console_scripts": [
            "starry-process=starry_process.app:entry_point [app]"
        ]
    },
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=False,
)
