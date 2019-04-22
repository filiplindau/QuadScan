# -*- coding: utf-8 -*-
"""
Created 2019-04-22

Test with scanning quads for constant on screen size

@author: Filip Lindau
"""

import numpy as np
import itertools
from collections import OrderedDict
import threading
import time
import scipy.optimize as so
from collections import namedtuple

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.INFO)


class LatticeElement(object):
    def __init__(self, pos):
        self.pos = pos

    def get_abcd(self, x=None):
        return np.identity(2)

    def get_pos(self):
        return self.pos


class Drift(LatticeElement):
    def __init__(self, l, pos):
        LatticeElement.__init__(self, pos)
        self.M = np.identity(2)
        self.M[0, 1] = l

    def get_abcd(self, x):
        return self.M


class Quad(LatticeElement):
    def __init__(self, l, pos):
        LatticeElement.__init__(self, pos)
        self.l = l

    def get_abcd(self, x):
        if x >= 0:
            sqrt_k = np.sqrt(x)
            return np.array([[np.cos(sqrt_k * self.l), np.sin(sqrt_k * self.l) / (sqrt_k + 1e-9)],
                            [-sqrt_k * np.sin(sqrt_k * self.l), np.cos(sqrt_k * self.l)]])
        else:
            sqrt_k = np.sqrt(-x)
            return np.array([[np.cosh(sqrt_k * self.l), np.sinh(sqrt_k * self.l) / sqrt_k],
                            [sqrt_k * np.sinh(sqrt_k * self.l), np.cosh(sqrt_k * self.l)]])


class ThinLens(LatticeElement):
    def __init__(self, l, pos):
        LatticeElement.__init__(self, pos)
        self.l = l

    def get_abcd(self, x):
        M = np.identity(2)
        M[1, 0] = -1.0/(np.sqrt(x)*self.l)
        return M


class Screen(LatticeElement):
    def __init__(self, pos):
        LatticeElement.__init__(self, pos)
        self.M = np.identity(2)

    def get_abcd(self, x):
        return self.M


class Lattice(object):
    def __init__(self):
        # self.magnet_dict = dict()
        self.magnet_list = list()
        self.drift_matrix_list = list()
        self.lattice = list()
        self.M = np.identity(2)

    def add_magnet(self, magnet):
        self.magnet_list.append(magnet)
        if len(self.magnet_list) > 1:
            l = self.magnet_list[-1].get_pos() - self.magnet_list[-2].get_pos()
            drift = Drift(l, magnet.get_pos())
            self.drift_matrix_list.append(drift.get_abcd(0))
            self.lattice.append(drift)
        self.lattice.append(magnet)

    def get_abcd(self, x0):
        M = self.magnet_list[0].get_abcd(x0[0])
        for ind, mag in enumerate(self.magnet_list[1:]):
            M = np.matmul(np.matmul(mag.get_abcd(x0[ind+1]), self.drift_matrix_list[ind]), M)
        return M


def f(x, a, b, lat):
    M = lat.get_abcd(x)
    Ml = ((M[0, 0] - a)**2 + (M[0, 1] - b)**2) / (np.abs(a)+np.abs(b))
    xl = np.mean(x**2)
    return 10*Ml+0.1*xl


if __name__ == "__main__":
    lat = Lattice()
    lat.add_magnet(Quad(0.2, 0.0))
    lat.add_magnet(Quad(0.2, 1.0))
    lat.add_magnet(Quad(0.2, 2.0))
    lat.add_magnet(Quad(0.2, 3.0))
    lat.add_magnet(Screen(7.0))

    lat.get_abcd([1.0, -1.0, 1.5, -0.5, 0.0])

    xm = so.minimize(f, [1.0, 1.0, -1.0, -1.0, 0], args=(1, 1, lat))


