import numpy as np
from scipy.special import gamma

__all__ = [
    "params",
    "ContrastPDF",
    "SizePDF",
    "LatitudePDF",
    "IdentityTransform",
    "AlphaBetaTransform",
]


params = {
    "latitude": {
        "mu": {"start": 0.01, "stop": 0.99, "step": 0.01, "value": 0.95},
        "nu": {"start": 0.01, "stop": 0.99, "step": 0.01, "value": 0.1},
    },
    "size": {
        "mu": {"start": 0.01, "stop": 0.99, "step": 0.01, "value": 0.1},
        "nu": {"start": 0.01, "stop": 0.99, "step": 0.01, "value": 0.1},
    },
    "contrast": {
        "mu": {"start": -5, "stop": 1, "step": 0.01, "value": 0.0},
        "nu": {"start": 0.01, "stop": 1.0, "step": 0.01, "value": 0.01},
    },
}


def IdentityTransform(*args):
    return args


def AlphaBetaTransform(mu, nu):
    alpha = mu * (1 / nu - 1)
    beta = (1 - mu) * (1 / nu - 1)
    return alpha, beta


def ContrastPDF(x, mu, nu):
    return (
        1.0
        / ((1 - x) * np.sqrt(2 * np.pi * nu))
        * np.exp(-((np.log(1 - x) - mu) ** 2) / (2 * nu))
    )


def SizePDF(x, mu, nu):
    alpha, beta = AlphaBetaTransform(mu, nu)
    return (
        gamma(alpha + beta)
        / (gamma(alpha) * gamma(beta))
        * x ** (alpha - 1)
        * (1 - x) ** (beta - 1)
    )


def LatitudePDF(x, mu, nu):
    alpha, beta = AlphaBetaTransform(mu, nu)
    return (
        gamma(alpha + beta)
        / (gamma(alpha) * gamma(beta))
        * 0.5
        * np.abs(np.sin(x * np.pi / 180))
        * np.cos(x * np.pi / 180) ** (alpha - 1)
        * (1 - np.cos(x * np.pi / 180)) ** (beta - 1)
    )
