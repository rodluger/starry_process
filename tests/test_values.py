from starry_gp.gp import YlmGP
import starry
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")


def test_infs():
    gp = YlmGP(10)

    fig, ax = plt.subplots(2)

    gp.set_params(0.81, 0.0007, -2.9, 0.30, -2.50, 0.50)
    ax[0].plot(np.diag(gp.cov))

    ax[1].plot(np.diag(gp.S.Q))

    gp.set_params(0.81, 0.0007, -2.9, 0.40, -2.50, 0.50)
    ax[0].plot(np.diag(gp.cov))

    ax[1].plot(np.diag(gp.S.Q))

    gp.set_params(0.81, 0.0007, -2.9, 0.45, -2.50, 0.50)
    ax[0].plot(np.diag(gp.cov))

    ax[1].plot(np.diag(gp.S.Q))

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")

    fig, ax = plt.subplots(2)
    gp.set_params(0.81, 0.0007, -2.9, 0.40, -2.50, 0.50)
    ax[0].imshow(gp.S.coeffs @ gp.S.Q)
    gp.set_params(0.81, 0.0007, -2.9, 0.45, -2.50, 0.50)
    ax[1].imshow(gp.S.coeffs @ gp.S.Q)

    # DEBUG
    s = np.linspace(0.35, 0.45, 100)
    v = np.zeros_like(s)
    for i in range(len(s)):
        gp.set_params(0.81, 0.0007, -2.9, s[i], -2.50, 0.50)
        v[i] = gp.S.U[-1, 4]
    plt.figure()
    plt.plot(s, v)
    plt.show()


if __name__ == "__main__":
    # test_infs()

    intensity = -0.1
    sigma = 0.6
    map = starry.Map(ydeg=10, lazy=False)
    map.add_spot(intensity=intensity, sigma=sigma)

    # Compute the intensity along the equator
    # Remove the baseline intensity and normalize it
    lon = np.linspace(-180, 180, 1000)
    baseline = 1.0 / np.pi
    I = -(map.intensity(lon=lon) - baseline) / (map.intensity(lon=0) - baseline)

    # Compute the intensity of a normalized gaussian
    # in cos(longitude) with the same standard deviation
    coslon = np.cos(lon * np.pi / 180)
    I_gaussian = -np.exp(-((coslon - 1) ** 2) / (2 * sigma ** 2))

    # Compare the two
    plt.plot(lon, I, label="starry ($l = 30$)")
    plt.plot(lon, I_gaussian, label="gaussian")
    plt.legend()
    plt.xlabel("longitude [deg]")
    plt.ylabel("normalized intensity")

    plt.show()
