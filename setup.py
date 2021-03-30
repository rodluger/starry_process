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
        "pymc3",
        "pymc3-ext",
        "aesara-theano-fallback>=0.0.4",
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
            "corner",
            "starry",
        ],
        "docs": [
            "sphinx>=1.7.5",
            "pandoc",
            "jupyter",
            "jupytext",
            "ipywidgets",
            "nbformat",
            "nbconvert",
            "corner",
            "emcee",
            "rtds_action",
            "nbsphinx",
            "starry",
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
