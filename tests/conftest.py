import pytest
import theano


@pytest.fixture(autouse=True)
def theano_setup(*args):
    theano.config.compute_test_value = "ignore"
    yield
