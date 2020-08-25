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
        "numpy>=1.13.0",
        "scipy>=1.2.1",
        "theano>=1.0.4",
    ],
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=False,
)
