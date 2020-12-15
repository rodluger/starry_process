import json
import os
from warnings import warn

PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULTS = os.path.join(PATH, "defaults.json")


def _update(inputs, defaults):
    """
    
    """
    # Update the `defaults` dict with entries in `inputs`
    for key, value in defaults.items():
        if key in inputs:
            if type(value) is dict:
                defaults[key] = _update(inputs[key], value)
            else:
                defaults[key] = inputs[key]

    # Check for invalid entries in `inputs`
    for key, value in inputs.items():
        if key not in defaults:
            warn("Invalid keyword: {}. Ignoring.".format(key))

    return defaults


def update_with_defaults(**kwargs):
    """
    Update kwargs with defaults (if values are missing).

    """
    # Update the defaults with the input values
    with open(DEFAULTS, "r") as f:
        defaults = json.load(f)
    return _update(kwargs, defaults)
