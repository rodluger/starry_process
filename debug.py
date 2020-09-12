import starry
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import theano.sparse as ts


def _func(res, theta):
    ydeg = 15
    map = starry.Map(ydeg, lazy=True)
    map.load("earth")

    # Compute the Cartesian grid
    xyz = map.ops.compute_moll_grid(res)[-1]

    # Compute the polynomial basis
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])

    # If orthographic, rotate the map to the correct frame
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))

    # Change basis to polynomials
    A1Ry = ts.dot(map.ops.A1, Ry)

    # Dot the polynomial into the basis
    res = tt.reshape(tt.dot(pT, A1Ry), [res, res, -1])

    # We need the shape to be (nframes, npix, npix)
    return res.dimshuffle(2, 0, 1)


_res = tt.iscalar()
_theta = tt.dvector()
func = theano.function([_res, _theta], _func(_res, _theta))

print(func(300, np.array([0.0])))
