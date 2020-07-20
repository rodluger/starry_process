from starry_gp.gp import YlmGP


def test_infs():
    gp = YlmGP(10)
    gp.set_params(0.87, 0.005, -1.10, 0.20, -2.50, 0.50)


if __name__ == "__main__":
    test_infs()
