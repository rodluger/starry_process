import numpy as np
from scipy.stats import beta as Beta


def pdf(phi, alpha, beta):
    """Latitude PDF as a function of latitude."""
    return 0.5 * np.abs(np.sin(phi)) * Beta.pdf(np.cos(phi), alpha, beta)


def deriv_pdf(phi, alpha, beta, eps=1e-8):
    """Derivative of `pdf` with respect to latitude."""
    return (pdf(phi + eps, alpha, beta) - pdf(phi - eps, alpha, beta)) / (
        2 * eps
    )


def log_pdf(phi, alpha, beta):
    """Natural log of `pdf`."""
    return np.log(0.5 * np.abs(np.sin(phi))) + Beta.logpdf(
        np.cos(phi), alpha, beta
    )


def second_deriv_log_pdf(phi, alpha, beta, eps=1e-4):
    """Second derivative of `log_pdf` with respect to latitude."""
    return (
        log_pdf(phi + eps, alpha, beta)
        - 2 * log_pdf(phi, alpha, beta)
        + log_pdf(phi - eps, alpha, beta)
    ) / eps ** 2


def get_mu(alpha, beta):
    """Compute the mode of the latitude distribution."""
    term = (
        4 * alpha ** 2
        - 8 * alpha
        - 6 * beta
        + 4 * alpha * beta
        + beta ** 2
        + 5
    )
    return 2 * np.arctan(np.sqrt(2 * alpha + beta - 2 - np.sqrt(term)))


def get_sigma(alpha, beta):
    """Compute the local standard deviation of the latitude distribution."""
    mu = get_mu(alpha, beta)
    term = (
        1
        - alpha
        + beta
        + (beta - 1) * np.cos(mu)
        + (alpha - 1) / np.cos(mu) ** 2
    )
    return np.sin(mu) / np.sqrt(term)


def get_alpha(mu, sigma):
    """Compute the Beta shape parameter #1."""
    v = sigma ** 2
    c1 = np.cos(mu)
    c2 = np.cos(2 * mu)
    c3 = np.cos(3 * mu)
    term = 1.0 / (16 * v * np.cos(0.5 * mu) ** 4)
    return (2 + 4 * v + (3 + 8 * v) * c1 + 2 * c2 + c3) * term


def get_beta(mu, sigma):
    """Compute the Beta shape parameter #2."""
    v = sigma ** 2
    c1 = np.cos(mu)
    c2 = np.cos(2 * mu)
    c3 = np.cos(3 * mu)
    term = 1.0 / (16 * v * np.cos(0.5 * mu) ** 4)
    return (c1 + 2 * v * (3 + c2) - c3) * term


def test_mu(ntests=100):
    """
    Show that the derivative of the PDF is zero at phi = mu;
    that is, we've found the peak (mode) of the distribution.

    """
    np.random.seed(0)
    for n in range(ntests):
        a = np.random.random()
        b = np.random.random()
        alpha = np.exp(a * 5)
        beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))
        mu = get_mu(alpha, beta)
        deriv = deriv_pdf(mu, alpha, beta)
        assert np.abs(deriv) < 1e-5


def test_sigma(ntests=100, ftol=1e-2):
    """
    Show that -1/sigma^2 is the curvature of the PDF at phi = mu.

    """
    np.random.seed(0)
    for n in range(ntests):
        a = np.random.random()
        b = np.random.random()
        alpha = np.exp(a * 5)
        beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))
        mu = get_mu(alpha, beta)
        sigma = get_sigma(alpha, beta)
        val1 = -1 / sigma ** 2
        val2 = second_deriv_log_pdf(mu, alpha, beta)
        assert np.abs((val1 - val2) / val1) < ftol


def test_alpha(ntests=100):
    """
    Show that the inverse transform alpha(mu, sigma) is correct.
    
    """
    np.random.seed(0)
    for n in range(ntests):
        a = np.random.random()
        b = np.random.random()
        alpha = np.exp(a * 5)
        beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))

        # Forward
        mu = get_mu(alpha, beta)
        sigma = get_sigma(alpha, beta)

        # Backward
        alpha_ = get_alpha(mu, sigma)

        assert np.allclose(alpha, alpha_)


def test_beta(ntests=100):
    """
    Show that the inverse transform beta(mu, sigma) is correct.
    
    """
    np.random.seed(0)
    for n in range(ntests):
        a = np.random.random()
        b = np.random.random()
        alpha = np.exp(a * 5)
        beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))

        # Forward
        mu = get_mu(alpha, beta)
        sigma = get_sigma(alpha, beta)

        # Backward
        beta_ = get_beta(mu, sigma)

        assert np.allclose(beta, beta_)
