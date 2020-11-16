import json
import os

PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULTS = os.path.join(PATH, "defaults.json")


def update_with_defaults(**kwargs):
    """
    Update kwargs with defaults (if values are missing).

    This is quite ugly & hacky. Supports up to two levels
    of nested dictionaries.
    
    """
    with open(DEFAULTS, "r") as f:
        defaults = json.load(f)
        for key0, value0 in defaults.items():
            if (type(value0) is dict) and key0 in kwargs:
                for key1, value1 in value0.items():
                    if (type(value1) is dict) and key1 in kwargs[key0]:
                        for key2, value2 in value1.items():
                            if type(value2) is dict:
                                raise NotImplementedError("")
                            if key2 in kwargs[key0][key1]:
                                defaults[key0][key1][key2] = kwargs[key0][
                                    key1
                                ][key2]
                    elif key1 in kwargs[key0]:
                        defaults[key0][key1] = kwargs[key0][key1]
            elif key0 in kwargs:
                defaults[key0] = kwargs[key0]
    return defaults
