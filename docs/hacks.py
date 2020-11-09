import starry_process
from starry_process.defaults import defaults
import sys
import os

# Add the CWD to the path
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# Populate the default values in the docstrings
for prop in [
    starry_process.StarryProcess,
    starry_process.beta2gauss,
    starry_process.gauss2beta,
]:
    for obj in [prop] + [getattr(prop, key) for key in prop.__dict__.keys()]:
        if hasattr(obj, "__doc__"):
            try:
                for key, value in defaults.items():
                    obj.__doc__ = obj.__doc__.replace(
                        '%%defaults["{}"]%%'.format(key),
                        "``{}``".format(value),
                    )
            except:
                pass

# Create some dummy functions for the various integrals
# so we can use ``.. autofunction::`` in the docs.


def latitude_pdf(phi):
    """
    Return the probability density function evaluated at
    latitudes ``phi``.

    Parameters:
        phi (vector): The latitudes (in degrees) at which to evaluate the PDF.
    """
    pass


starry_process.latitude.pdf = latitude_pdf


def latitude_sample(nsamples=1):
    """
    Draw samples from the latitude distribution.

    Parameters:
        nsamples (int, optional): The number of samples to return. Default is 1.
    """
    pass


starry_process.latitude.sample = latitude_sample


def longitude_pdf(lam):
    """
    Return the probability density function evaluated at
    longitudes ``lam``.

    Parameters:
        lam (vector): The longitudes (in degrees) at which to evaluate the PDF.
    """
    pass


starry_process.longitude.pdf = longitude_pdf


def longitude_sample(nsamples=1):
    """
    Draw samples from the longitude distribution.

    Parameters:
        nsamples (int, optional): The number of samples to return. Default is 1.
    """
    pass


starry_process.longitude.sample = longitude_sample
