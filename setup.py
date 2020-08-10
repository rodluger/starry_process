"""Install script for `starry_gp`."""
from setuptools import setup

setup(
    name="starry_gp",
    version="0.0.1",
    author="Rodrigo Luger",
    author_email="rodluger@gmail.com",
    url="https://github.com/rodluger/starry_gp",
    description="starry gaussian processes",
    license="MIT",
    packages=["starry_gp"],
    install_requires=["scipy>=1.2.1", "starry>=1.0.0"],
    include_package_data=True,
    zip_safe=False,
)
