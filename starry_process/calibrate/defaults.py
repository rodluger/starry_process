import json
import os
from warnings import warn

PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULTS = os.path.join(PATH, "defaults.json")


def _update(inputs, defaults, ignore_invalid=True):
    """
    
    """
    # Update the `defaults` dict with entries in `inputs`
    for key, value in defaults.items():
        if key in inputs:
            if type(value) is dict:
                # Allow anything in these dicts (passed directly to dynesty)
                if key in ["run_nested_kwargs", "sampler_kwargs"]:
                    defaults[key] = _update(inputs[key], value, False)
                else:
                    defaults[key] = _update(inputs[key], value)
            else:
                defaults[key] = inputs[key]

    # Check for invalid entries in `inputs`
    for key, value in inputs.items():
        if key not in defaults:
            if ignore_invalid:
                warn("Invalid keyword: {}. Ignoring.".format(key))
            else:
                defaults[key] = value
    return defaults


def update_with_defaults(**kwargs):
    """
    Update kwargs with defaults (if values are missing).

    """
    # Update the defaults with the input values
    with open(DEFAULTS, "r") as f:
        defaults = json.load(f)
    return _update(kwargs, defaults)
