"""
Created 2020-01-22

Simulation of spot size for quad settings given transfer matrix and input twiss parameters

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


try:
    import PyTango as pt
except ImportError:
    try:
        import tango as pt
    except ModuleNotFoundError:
        pass


class QuadSimulator(object):
    def __init__(self, alpha, beta, eps):
        self.name = "QuadSimulator"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))
        self.logger.setLevel(logging.INFO)

        self.alpha = None
        self.beta = None
        self.eps = None
        self.sigma1 = None

        self.set_start_twiss_params(alpha, beta, eps)

        self.quad_list = list()
        self.quad_strengths = np.array([])
        self.screen = SectionScreen("screen", 0, "test/screen/liveviewer", "test/screen/beamviewer",
                                    "test/screen/limaccd", "test/screen/screen")

    def add_quad(self, quad, quad_strength=0.0):
        self.logger.info("Adding quad at position {0}".format(quad.position))
        self.quad_list.append(quad)
        pos_array = np.array([q.position for q in self.quad_list])
        ind_s = np.argsort(pos_array)
        tmp_list = [self.quad_list[ind] for ind in ind_s]
        self.quad_list = tmp_list
        self.quad_strengths = np.hstack((self.quad_strengths, quad_strength))[ind_s]

    def remove_quad(self, index):
        self.logger.info("Removing quad at index {0}".format(index))
        self.quad_list.pop(index)

    def set_screen_position(self, position):
        self.logger.debug("Setting screen position to {0}".format(position))
        self.screen = SectionScreen(self.screen.name, position, self.screen.liveviewer, self.screen.beamviewer,
                                    self.screen.limaccd, self.screen.screen)

    def set_quad_strength(self, k):
        """
        Set quad strenth array to k
        :param k: Numpy array with same length as the number of quads
        :return:
        """
        self.logger.debug("Setting quads {0}".format(k))
        self.quad_strengths = k

    def set_start_twiss_params(self, alpha, beta, eps):
        self.logger.info("New Twiss parameters: alpha={0}, beta={1}, eps={2}".format(alpha, beta, eps))
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        gamma = (1.0 + alpha**2) / beta
        self.sigma1 = eps * np.array([[beta, -alpha], [-alpha, gamma]])

    def calc_response_matrix(self, quad_strengths):
        self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = self.quad_list[0].position
        M = np.identity(2)
        for ind, quad in enumerate(self.quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = quad.length
            k = quad_strengths[ind]
            k_sqrt = np.sqrt(k*(1+0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sinc(k_sqrt * L) * L],
                                    [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            M = np.matmul(M_q, M)
            s = quad.position
        drift = self.screen.position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        return M

    def get_screen_twiss(self):
        M = self.calc_response_matrix(self.quad_strengths)
        sigma2 = np.matmul(np.matmul(M, self.sigma1), M.transpose())
        eps = np.sqrt(np.linalg.det(sigma2))
        alpha = -sigma2[0, 1] / eps
        beta = sigma2[0, 0] / eps
        return alpha, beta, eps

    def get_screen_beamsize(self, quad_strengths=None):
        if quad_strengths is None:
            M = self.calc_response_matrix(self.quad_strengths)
        else:
            M = self.calc_response_matrix(quad_strengths)
        a = M[0, 0]
        b = M[0, 1]
        sigma = np.sqrt(self.eps * (self.beta * a**2 - 2.0 * self.alpha * a * b + (1.0 + self.alpha**2) / self.beta * b**2))
        return sigma


class MultiQuad(object):
    def __init__(self):
        self.name = "QuadSimulator"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))

        self.gamma_energy = 241e6 / 0.511e6

        # self.sim = QuadSimulator(36.0, 69.0, 3e-6 * self.gamma_energy)  # MS-1 QB-01, SCRN-01
        self.sim = QuadSimulator(36.0, 69.0, 3e-6)  # MS-1 QB-01, SCRN-01
        self.sim.add_quad(SectionQuad("QB-01", 13.55, 0.2, "MAG-01", "CRQ-01", True))
        self.sim.add_quad(SectionQuad("QB-02", 14.45, 0.2, "MAG-02", "CRQ-02", True))
        self.sim.add_quad(SectionQuad("QB-03", 17.75, 0.2, "MAG-03", "CRQ-03", True))
        self.sim.add_quad(SectionQuad("QB-04", 18.65, 0.2, "MAG-04", "CRQ-04", True))
        self.sim.set_screen_position(19.223)

        self.quad_list = list()
        self.quad_list.append(SectionQuad("QB-01", 13.55, 0.2, "MAG-01", "CRQ-01", True))
        self.quad_list.append(SectionQuad("QB-02", 14.45, 0.2, "MAG-02", "CRQ-02", True))
        self.quad_list.append(SectionQuad("QB-03", 17.75, 0.2, "MAG-03", "CRQ-03", True))
        self.quad_list.append(SectionQuad("QB-04", 18.65, 0.2, "MAG-04", "CRQ-04", True))


        # Starting guess
        self.quad_strength_list = [-1.0, 0.0, 1.5, 1.5]

        self.screen = SectionScreen("screen", 19.223, "liveviewer", "beamviewer", "limaccd", "screen")

        self.algo = "const_size"

        self.a_list = list()
        self.b_list = list()
        self.M_list = list()
        self.k_list = list()
        self.x_list = list()
        self.psi_target = None

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.theta_list = list()
        self.r_maj_list = list()
        self.r_min_list = list()

        self.logger.setLevel(logging.DEBUG)

        self.scan()

    def scan(self):
        M0 = self.calc_response_matrix(self.quad_strength_list, self.quad_list, self.screen.position)
        a0 = M0[0, 0]
        b0 = M0[0, 1]
        sigma0 = self.sim.get_screen_beamsize(self.quad_strength_list)
        alpha0 = 0.0
        eps0 = 3e-6
        beta0 = self.get_missing_twiss(sigma0, M0, alpha0, None, eps0)
        theta, r_maj, r_min = self.calc_ellipse(alpha0, beta0, eps0, sigma0)
        psi0 = np.arcsin(a0 / np.sqrt(a0**2 + b0**2)) - theta

        self.logger.info("Scan starting. Initial parameters: th={0:.3f}, r_maj={1:.3f}, r_min={2:.3f}, "
                         "a0={3:.3f}, b0={4:.3f}".format(theta, r_maj, r_min, a0, b0))

        n_steps = 50
        self.psi_target = np.linspace(0, 1, n_steps) * 2 * np.pi / n_steps + psi0

        # for step in range(n_steps-1):
        for step in range(5):
            psi = self.psi_target[step + 1]
            a, b = self.set_target_ab(psi, theta, r_maj, r_min)
            sigma = self.scan_step(a, b)
            alpha, beta, eps = self.calc_twiss(np.array(self.a_list), np.array(self.b_list), np.array(self.x_list))
            theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps, sigma)
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            self.eps_list.append(eps)
            self.theta_list.append(theta)
            self.r_maj_list.append(r_maj)
            self.r_min_list.append(r_min)

    def scan_step(self, a, b):
        self.logger.info("{0} Scan step {1:.3f}, {2:.3f}".format(self, a, b))
        r = self.solve_quads(a, b)
        self.k_list.append(r.x)
        M = self.calc_response_matrix(r.x, self.quad_list, self.screen.position)
        self.a_list.append(M[0, 0])
        self.b_list.append(M[0, 1])
        sigma = self.sim.get_screen_beamsize(r.x)
        self.x_list.append(sigma)
        return sigma

    def calc_twiss(self, a, b, sigma):
        M = np.vstack((a*a, -2*a*b, b*b)).transpose()
        try:
            l_data = np.linalg.lstsq(M, sigma**2, rcond=-1)
            x = l_data[0]
            res = l_data[1]
        except Exception as e:
            self.logger.error("Error when fitting lstsqr: {0}".format(e))
            return e
        self.logger.debug("Fit coefficients: {0}".format(x[0]))
        eps = np.sqrt(x[2] * x[0] - x[1]**2)
        eps_n = eps * self.gamma_energy
        beta = x[0] / eps
        alpha = x[1] / eps
        return alpha, beta, eps

    def set_target_ab(self, psi, theta, r_maj, r_min):
        self.logger.info("{0}: Determine new target a,b for algo {1}".format(self, self.algo))
        if self.algo == "const_size":
            target_a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
            target_b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
        else:
            target_a = 1
            target_b = 0
        self.logger.debug("Target a, b = {0:.3f}, {1:.3f}".format(target_a, target_b))
        return target_a, target_b

    def solve_quads(self, target_a, target_b):
        self.logger.info("{0}: Solving new quad strengths for target a,b = {1:.2f}, {2:.2f}".format(self,
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
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = quad_list[0].position
        M = np.identity(2)
        for ind, quad in enumerate(quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = quad.length
            k = quad_strengths[ind]
            k_sqrt = np.sqrt(k*(1+0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sinc(k_sqrt * L) * L],
                                    [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            M = np.matmul(M_q, M)
            s = quad.position
        drift = screen_position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        return M

    def calc_ellipse(self, alpha, beta, eps, sigma):
        my = sigma**2.0 / eps
        gamma = (1.0 + alpha**2) / beta
        try:
            theta = np.arctan(2.0 * alpha / (gamma - beta)) / 2.0    # Ellipse angle
        except ZeroDivisionError:
            theta = np.pi/2
        m11 = beta
        m12 = -alpha
        m22 = gamma
        l1 = ((m11 + m22) + np.sqrt((m11 - m22) ** 2 + 4.0 * m12 ** 2)) / 2
        l2 = ((m11 + m22) - np.sqrt((m11 - m22) ** 2 + 4.0 * m12 ** 2)) / 2
        r_minor = np.sqrt(my / l1)
        r_major = np.sqrt(my / l2)
        theta = np.arctan((l1 - gamma) / alpha)
        return theta, r_major, r_minor

    def get_missing_twiss(self, sigma, M, alpha=None, beta=None, eps=None):
        a = M[0, 0]
        b = M[0, 1]
        # sigma = np.sqrt(self.eps * (self.beta * a**2 - 2.0 * self.alpha * a * b + (1.0 + self.alpha**2) / self.beta * b**2))
        if eps is None:
            res = sigma**2 / (beta * a**2 - 2.0 * alpha * a * b + (1.0 + alpha**2) / beta * b**2)
        elif beta is None:
            p = b * alpha / a + sigma**2 / (2 * eps * a**2)
            res = p + np.sqrt(p**2 - b**2 * (1 + alpha**2) / a**2)
        else:
            res = a * beta / b - np.sqrt(sigma**2 * beta / (b**2 * eps) - 1)
        return res

    def __repr__(self):
        s = "{0} {1}".format(type(self), self.name)
        return s

    def __str__(self):
        s = "{0} {1}".format(type(self).__name__, self.name)
        return s


if __name__ == "__main__":
    # -----MS1-----
    # QB-01: l=0.2, s=13.55
    # QB-02: l=0.2, s=14.45
    # QB-03: l=0.2, s=17.75
    # QB-04: l=0.2, s=18.65
    # Screen: s=19.223 m
    # sim = QuadSimulator(36, 69, 3e-6 * 241 / 0.511)       # MS-1 QB-01, SCRN-01
    sim = QuadSimulator(36, 69, 3e-6)  # MS-1 QB-01, SCRN-01
    # sim = QuadSimulator(0, 10, 3e-6 )  # MS-1 QB-01, SCRN-01
    sim.add_quad(SectionQuad("QB-01", 13.55, 0.2, "MAG-01", "CRQ-01", True))
    sim.add_quad(SectionQuad("QB-02", 14.45, 0.2, "MAG-02", "CRQ-02", True))
    sim.add_quad(SectionQuad("QB-03", 17.75, 0.2, "MAG-03", "CRQ-03", True))
    sim.add_quad(SectionQuad("QB-04", 18.65, 0.2, "MAG-04", "CRQ-04", True))
    sim.set_screen_position(19.223)
    # sim = QuadSimulator(19, 42, 2e-6)       # MS-1 QB-02, SCRN-01

    # sim.add_quad(SectionQuad("QB-01", 0, 0.2, "MAG-01", "CRQ-01", True))
    # sim.set_screen_position(5)

    print("Beam size: {0}".format(sim.get_screen_beamsize()))

    # k = np.linspace(-5, 5, 5000)
    # sigma = list()
    # t0 = time.time()
    # for kp in k:
    #     sim.set_quad_strength([kp, 0, 0, 0])
    #     sigma.append(sim.get_screen_beamsize())
    # t1 = time.time()
    # sigma = np.array(sigma)
    # logger.info("Time spent: {0}".format(t1-t0))

    mq = MultiQuad()

