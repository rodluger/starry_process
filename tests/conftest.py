import pytest
from starry_process.compat import theano


@pytest.fixture(autouse=True)
def theano_setup(*args):
    theano.config.compute_test_value = "ignore"
    yield


@pytest.fixture(autouse=True)
def starry_setup(*args):
    try:
        import starry
    except ImportError:
        pass
    else:
        starry.config.lazy = False
    yield
