from scipy.stats import beta as Beta
import numpy as np


def test_beta_transform(alpha=3.0, beta=5.0, nsamples=10000000, tol=1e-2):
    """
    Numericaly show that our variable transformation from `cos(phi)` to `phi`
    in the latitude PDF is correct.

    """
    # Draw from the PDF in `cos(phi)`
    cosphi = Beta.rvs(alpha, beta, size=nsamples)

    # Transform the samples to `+/-phi`
    sign = np.sign(np.random.random(nsamples) - 0.5)
    phi_samples = sign * np.arccos(cosphi)

    # Compute the empirical PDF
    bin_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    hist, _ = np.histogram(phi_samples, bins=bin_edges, density=True)

    # Compute the exact PDF in the same bins
    phi = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    pdf = 0.5 * np.abs(np.sin(phi)) * Beta.pdf(np.cos(phi), alpha, beta)

    # Check that they agree
    assert np.max(np.abs(hist - pdf)) < tol
