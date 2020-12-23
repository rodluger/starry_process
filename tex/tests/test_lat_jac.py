import numpy as np


def get_mu(a, b):
    """Compute the mode of the latitude distribution."""
    alpha = np.exp(a * 5)
    beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))
    term = (
        4 * alpha ** 2
        - 8 * alpha
        - 6 * beta
        + 4 * alpha * beta
        + beta ** 2
        + 5
    )
    return 2 * np.arctan(np.sqrt(2 * alpha + beta - 2 - np.sqrt(term)))


def get_sigma(a, b):
    """Compute the local standard deviation of the latitude distribution."""
    alpha = np.exp(a * 5)
    beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))
    mu = get_mu(a, b)
    term = (
        1
        - alpha
        + beta
        + (beta - 1) * np.cos(mu)
        + (alpha - 1) / np.cos(mu) ** 2
    )
    return np.sin(mu) / np.sqrt(term)


def get_J(a, b):
    """
    Compute the Jacobian for the transformation (a, b) <--> (mu, sigma).

    """
    alpha = np.exp(a * 5)
    beta = np.exp(np.log(0.5) + b * (5 - np.log(0.5)))
    mu = get_mu(a, b)
    sigma = get_sigma(a, b)
    C = 5 * (5 - np.log(0.5))
    return (
        C
        * (alpha * beta * (1 + np.cos(mu)) ** 3 * np.sin(2 * mu) ** 3)
        / (
            sigma
            * (-3 + 2 * alpha + beta + (-1 + 2 * alpha + beta) * np.cos(mu))
            * (
                2 * (-1 + alpha + beta)
                + 3 * (-1 + beta) * np.cos(mu)
                - 2 * (-1 + alpha - beta) * np.cos(2 * mu)
                + (-1 + beta) * np.cos(3 * mu)
            )
            ** 2
        )
    )


def test_jacobian(ntests=100, eps=1e-6):
    """
    Numerically show that our expression for the Jacobian agrees
    with its definition in terms of derivatives.
    
    """
    np.random.seed(0)

    for n in range(ntests):
        a = np.random.random()
        b = np.random.random()

        # Analytic expression
        J = get_J(a, b)

        # Numerical expression
        dmuda = (get_mu(a + eps, b) - get_mu(a - eps, b)) / (2 * eps)
        dmudb = (get_mu(a, b + eps) - get_mu(a, b - eps)) / (2 * eps)
        dsigmada = (get_sigma(a + eps, b) - get_sigma(a - eps, b)) / (2 * eps)
        dsigmadb = (get_sigma(a, b + eps) - get_sigma(a, b - eps)) / (2 * eps)
        J_num = dmuda * dsigmadb - dmudb * dsigmada

        assert np.allclose(J, J_num)
