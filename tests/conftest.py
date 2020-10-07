import pytest
import theano
import starry


@pytest.fixture(autouse=True)
def theano_setup(*args):
    theano.config.compute_test_value = "ignore"
    yield


@pytest.fixture(autouse=True)
def starry_setup(*args):
    starry.config.lazy = False
    yield
