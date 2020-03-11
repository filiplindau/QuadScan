"""
Created 2020-03-11

Calculation of ab lookup table

@author: Filip Lindau
"""

import threading
import multiprocessing
import uuid
import logging
import time
import ctypes
import inspect
import PIL
import numpy as np
from numpngw import write_png
import os
from collections import namedtuple
import pprint
import traceback
from scipy.signal import medfilt2d
from scipy.optimize import minimize, BFGS, NonlinearConstraint, Bounds, least_squares
from scipy.optimize import lsq_linear
from QuadScanTasks import TangoReadAttributeTask, TangoMonitorAttributeTask, TangoWriteAttributeTask, work_func_local2
import logging
from operator import attrgetter

from tasks.GenericTasks import *
from QuadScanDataStructs import *


import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


def calc_response_matrix(self, quad_strengths, quad_positions, screen_position, axis="x"):
    # self.logger.debug("{0}: Calculating new response matrix".format(self))
    s = quad_positions
    M = np.identity(2)
    if axis != "x":
        quad_strengths = -np.array(quad_strengths)
    for ind, quad in enumerate(quad_positions):
        # self.logger.debug("Position s: {0} m".format(s))
        drift = quad - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        L = quad.length
        k = quad_strengths[ind]
        if k != 0:
            k_sqrt = np.sqrt(k * (1 + 0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L), np.sin(k_sqrt * L) / k_sqrt],
                                    [-k_sqrt * np.sin(k_sqrt * L), np.cos(k_sqrt * L)]]))
        else:
            M_q = np.array([[1, L], [0, 1]])
        M = np.matmul(M_q, M)
        s = quad + L
    drift = screen_position - s
    M_d = np.array([[1.0, drift], [0.0, 1.0]])
    M = np.matmul(M_d, M)
    return M


section = "MS1"
if section == "MS1":
    screen_position = 19.223
    quad_positions = [13.55, 14.45, 17.75, 18.65]


