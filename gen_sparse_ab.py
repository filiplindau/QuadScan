"""
Created 2020-04-03

Generate a sparse ab lookup data structure from a saved data file.
The input data is generated by MultiQuadLookup.generate_lookup

@author: Filip Lindau
"""

import time
import multiprocessing as mp
import numpy as np
from numpngw import write_png
import os
import json
from collections import namedtuple, OrderedDict
import pprint
import traceback
from scipy.signal import medfilt2d
from scipy.optimize import minimize, leastsq # Bounds, least_squares, BFGS, NonlinearConstraint
#from scipy.optimize import lsq_linear
# from QuadScanTasks import TangoReadAttributeTask, TangoMonitorAttributeTask, TangoWriteAttributeTask, work_func_local2
from operator import attrgetter

#from tasks.GenericTasks import *
from QuadScanDataStructs import *


import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def generate_sparse_ab(data_filename="MS1_big.npz"):
    logger.info("Loading data...")
    npzfile = np.load(data_filename)
    A_lu = npzfile["A_lu"]
    k_lu = npzfile["k_lu"]
    logger.info("Loading complete. Starting binning")
    da = 0.2
    a_v = np.arange(-2, 2, da)
    db = 0.2
    b_v = np.arange(5, 7, db)
    A_dict = dict()
    for a_x in a_v:
        for b_x in b_v:
            logger.debug("a_x {0:.1f}, b_x {1:.1f}".format(a_x, b_x))
            a_ind = np.logical_and(A_lu[:, 0] > a_x - da / 2, A_lu[:, 0] < a_x + da / 2)
            b_ind = np.logical_and(A_lu[:, 1] > b_x - db / 2, A_lu[:, 1] < b_x + db / 2)
            ind = np.logical_and(a_ind, b_ind)
            a_y_p = A_lu[ind, 2]
            b_y_p = A_lu[ind, 3]
            k_tmp = k_lu[ind, :]
            y_dict_tmp = dict()
            for a_y in a_v:
                for b_y in b_v:
                    a_ind = np.logical_and(a_y_p > a_y - da / 2, a_y_p < a_y + da / 2)
                    b_ind = np.logical_and(b_y_p > b_y - db / 2, b_y_p < b_y + db / 2)
                    ind_y = np.logical_and(a_ind, b_ind)
                    k_good = k_tmp[ind_y, :]
                    if k_good.shape[0] > 0:
                        y_dict_tmp[(a_y, b_y)] = k_good
            if len(y_dict_tmp) > 0:
                A_dict[(a_x, b_x)] = y_dict_tmp
    return A_dict