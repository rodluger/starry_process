from starry_process import StarryProcess
import numpy as np


def test_variance():

    sp = StarryProcess(normalized=False)
    cov = sp.cov([0.0, 0.1]).eval()
    var = sp.cov([0.0]).eval()

    assert np.allclose(cov[0, 0], var)
