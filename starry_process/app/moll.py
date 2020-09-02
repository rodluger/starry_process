# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["get_latitude_lines", "get_longitude_lines"]


def get_latitude_lines(dlat=np.pi / 6, npts=1000, niter=100):
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:
        theta = lat
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = np.linspace(-2, 2, npts)
        y = np.ones(npts) * np.sin(theta)
        a = 1
        b = 2
        y[(y / a) ** 2 + (x / b) ** 2 > 1] = np.nan
        res.append((x, y))
    return res


def get_longitude_lines(dlon=np.pi / 6, npts=1000, niter=100):
    res = []
    lonlines = np.arange(-np.pi, np.pi, dlon)[1:]
    for lon in lonlines:
        lat = np.linspace(-np.pi / 2, np.pi / 2, npts)
        theta = np.array(lat)
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = 2 / np.pi * lon * np.cos(theta)
        y = np.sin(theta)
        res.append((x, y))
    return res
