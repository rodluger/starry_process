import starry
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import theano.sparse as ts
from starry.maps import logger


ydeg = 15
map = starry.Map(ydeg, lazy=True)
map.load("earth")
_res = tt.iscalar()
_theta = tt.dvector()
logger.info("starting")


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    return xyz


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("1 ok")


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    return pT


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("2 ok")


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))
    return Ry


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("3 ok")


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))
    A1Ry = ts.dot(map.ops.A1, Ry)
    return A1Ry


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("4 ok")


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))
    A1Ry = ts.dot(map.ops.A1, Ry)
    res = tt.dot(pT, A1Ry)
    return res


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("5 ok")
logger.info(x.shape)


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))
    A1Ry = ts.dot(map.ops.A1, Ry)
    res = tt.dot(pT, A1Ry)
    return res.shape


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("5 ok")
logger.info(x)


def _func(res, theta):
    xyz = map.ops.compute_moll_grid(res)[-1]
    pT = map.ops.pT(xyz[0], xyz[1], xyz[2])
    Ry = tt.transpose(tt.tile(map.y, [theta.shape[0], 1]))
    A1Ry = ts.dot(map.ops.A1, Ry)
    res = tt.reshape(tt.dot(pT, A1Ry), [res, res, -1])
    return res


func = theano.function(
    [_res, _theta], _func(_res, _theta), on_unused_input="ignore"
)
x = func(300, np.array([0.0]))
logger.info("6 ok")
