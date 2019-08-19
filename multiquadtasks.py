"""
Created 2018-10-24

Tasks for async sequencing.

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
from scipy.optimize import minimize
from QuadScanTasks import TangoReadAttributeTask, TangoMonitorAttributeTask, TangoWriteAttributeTask, work_func_local2

from tasks.GenericTasks import *
from QuadScanDataStructs import *


try:
    import PyTango as pt
except ImportError:
    try:
        import tango as pt
    except ModuleNotFoundError:
        pass


class MultiQuadScanTask(Task):
    def __init__(self, scan_param=None, device_handler=None, name=None, timeout=None, trigger_dict=dict(),
                 callback_list=list(), read_callback=None):
        # type: (ScanParam) -> None
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.scan_param = scan_param
        self.device_handler = device_handler
        self.scan_result = None
        self.last_step_result = None
        self.read_callback = read_callback

        self.quad_list = list()
        self.quad_strength_list = list()
        self.screen = None
        self.current_fit = None
        self.ab_list = list()
        self.algo = "const_size"
        self.pos_ind = 0

        self.logger.debug("{0}: Scan parameters: {1}".format(self, scan_param))
        self.logger.debug("{0}: read_callback {1}".format(self, read_callback))
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} starting scan of {1} from {2} to {3}. ".format(self, self.scan_param.scan_attr_name,
                                                                             self.scan_param.scan_start_pos,
                                                                             self.scan_param.scan_end_pos))
        self.logger.info("{0} measuring {1}. ".format(self, self.scan_param.measure_attr_name_list))
        pos_list = list()
        pos_ind = 0
        timestamp_list = list()
        meas_list = list()
        next_pos = self.scan_param.scan_start_pos
        while self.get_done_event().is_set() is False:
            write_pos_task = TangoWriteAttributeTask(self.scan_param.scan_attr_name,
                                                     self.scan_param.scan_device_name,
                                                     self.device_handler,
                                                     next_pos,
                                                     name="write_pos",
                                                     timeout=self.timeout)
            monitor_pos_task = TangoMonitorAttributeTask(self.scan_param.scan_attr_name,
                                                         self.scan_param.scan_device_name,
                                                         self.device_handler,
                                                         next_pos,
                                                         tolerance=self.scan_param.scan_pos_tol,
                                                         interval=self.scan_param.scan_pos_check_interval,
                                                         name="monitor_pos",
                                                         timeout=self.timeout)
            measure_task_list = list()
            for meas_ind, meas_attr in enumerate(self.scan_param.measure_attr_name_list):
                # m_name = "read_{0}_{1}_{2}".format(meas_attr, pos_ind, self.last_step_result)
                m_name = "read_{0}_{1}_{2}_{3}".format(meas_attr, pos_ind, next_pos, meas_ind)
                self.logger.info("Measuring {0} on {1} using device handler {2}".format(meas_attr,
                                                                                        self.scan_param.measure_device_list[meas_ind],
                                                                                        self.device_handler))
                if self.read_callback is None:
                    read_task = TangoReadAttributeTask(meas_attr, self.scan_param.measure_device_list[meas_ind],
                                                       self.device_handler, name=m_name,
                                                       timeout=self.timeout)
                else:
                    self.logger.info("{0}: Reading {1}".format(self, m_name))
                    read_task = TangoReadAttributeTask(meas_attr, self.scan_param.measure_device_list[meas_ind],
                                                       self.device_handler, name=m_name,
                                                       timeout=self.timeout, callback_list=[self.read_callback])
                rep_task = RepeatTask(read_task, self.scan_param.measure_number, self.scan_param.measure_interval,
                                      name="rep_{0}".format(meas_attr), timeout=self.timeout)
                measure_task_list.append(rep_task)
            measure_bag_task = BagOfTasksTask(measure_task_list, name="measure_bag", timeout=self.timeout)
            step_sequence_task = SequenceTask([write_pos_task, monitor_pos_task, measure_bag_task], name="step_seq")
            step_sequence_task.start()
            step_result = step_sequence_task.get_result(wait=True, timeout=self.timeout)
            if step_sequence_task.is_cancelled() is True:
                self.cancel()
                return
            if self.is_cancelled() is True:
                return
            pos_list.append(step_result[1].value)
            timestamp_list.append(step_result[1].time)
            meas_list.append(step_result[2])
            self.last_step_result = step_result
            self.result = step_result
            # Step done, notify callbacks:
            if self.is_done() is False:
                self.logger.debug("{0} Calling {1} callbacks".format(self, len(self.callback_list)))
                for callback in self.callback_list:
                    callback(self)
            next_pos += self.scan_param.scan_step
            pos_ind += 1
            if next_pos > self.scan_param.scan_end_pos:
                self.event_done.set()

        self.scan_result = ScanResult(pos_list, meas_list, timestamp_list)
        self.result = self.scan_result

    def get_last_step_result(self):
        return self.last_step_result

    def set_target_ab(self):
        self.logger.info("{0}: Determine new target a,b for algo {1}".format(self, self.algo))
        if self.algo == "const_size":
            target_a = 1
            target_b = 0
        else:
            target_a = 1
            target_b = 0
        return target_a, target_b

    def solve_quads(self, target_a, target_b):
        self.logger.info("{0}: Solving new quad strengts for target a,b = {1:.2f}, {2:.2f}".format(self,
                                                                                                   target_a,
                                                                                                   target_b))
        x0 = self.quad_strength_list
        res = minimize(self.opt_fun, x0=x0, args=[target_a, target_b], bounds=((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
        return res

    def opt_fun(self, x, target_ab):
        M = self.calc_response_matrix(x, self.quad_list, self.screen.position)
        y = (M[0, 0] - target_ab[0])**2 + (M[0, 1] - target_ab[1])**2
        return y

    def calc_response_matrix(self, quad_strengths, quad_list, screen_position):
        self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = quad_list[0].position
        M = np.identity(2)
        for ind, quad in enumerate(quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M, M_d)
            L = quad.length
            k = quad_strengths[ind]
            k_sqrt = np.sqrt(k*(1+0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sinc(k_sqrt * L) * L],
                                    [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            M = np.matmul(M, M_q)
            s = quad.position
        drift = screen_position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M, M_d)
        return M

    def process_image(self, image):
        work_func_local2(image)

    def save_image(self):
        pass

    def read_images(self):
        pass

    def write_quads(self):
        pass

    def calc_ellipse(self, alpha, beta, eps, sigma):
        my = sigma**2 / eps
        gamma = (1 + alpha**2) / beta
        try:
            theta = np.arctan(2*alpha / (gamma - beta))     # Ellipse angle
        except ZeroDivisionError:
            theta = np.pi/2
        m11 = my * beta
        m12 = -my * alpha
        m22 = my * gamma
        l1 = ((m11 + m22) + np.sqrt((m11 - m22) ** 2 + 4 * m12 **2 )) / 2
        l2 = ((m11 + m22) - np.sqrt((m11 - m22) ** 2 + 4 * m12 ** 2)) / 2
        r_minor = 1.0 / np.sqrt(l1)
        r_major = 1.0 / np.sqrt(l2)
        return theta, r_major, r_minor


if __name__ == "__main__":
    mq = MultiQuadScanTask()

    # MS1
    Q1 = SectionQuad("QB1", 13.55, 0.2, "i-ms1/mag/qb-01", "i-ms1/mag/crq-01", 1)
    Q2 = SectionQuad("QB2", 14.45, 0.2, "i-ms1/mag/qb-02", "i-ms1/mag/crq-02", 1)
    Q3 = SectionQuad("QB3", 17.75, 0.2, "i-ms1/mag/qb-03", "i-ms1/mag/crq-03", 1)
    Q4 = SectionQuad("QB4", 18.65, 0.2, "i-ms1/mag/qb-04", "i-ms1/mag/crq-04", 1)
    screen = SectionScreen("SCRN-01", 19.223, "lima/liveviewer/i-ms1-dia-scrn-01",
                           "lima/beamviewer/i-ms1-dia-scrn-01", "lima/limaccd/i-ms1-dia-scrn-01", "i-ms1/dia/scrn-01")
    mq.quad_list = [Q1, Q2, Q3, Q4]
    mq.quad_strength_list = [0, 0, 0, 0]
    mq.screen = screen
    alpha = -0.7
    beta = 9.67
    eps = 1.67e-6
    sigma = 100e-6
