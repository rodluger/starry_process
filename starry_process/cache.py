from pathlib import Path
import numpy as np
import functools


def cache(*cache_args):
    """
    Decorator for caching pre-computable things. Not
    currently used in the code.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, clobber=False, **kwargs):
            try:
                CACHE_DIR = Path.home() / ".starry_process"
                CACHE_DIR.mkdir(exist_ok=True)
                classname = str(self.__class__).split("'")[1]
                file = CACHE_DIR / (
                    "{}.{}-".format(classname, func.__name__)
                    + "-".join(
                        ["{}".format(getattr(self, arg)) for arg in cache_args]
                    )
                    + ".npz"
                )
                if (not clobber) and file.exists():
                    return np.load(str(file))
                else:
                    results = func(self, *args, **kwargs)
                    np.savez(file, **results)
                    return results
            except:
                # caching failed!
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
