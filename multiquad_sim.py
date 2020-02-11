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
from scipy.optimize import minimize, BFGS, NonlinearConstraint, Bounds
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
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = self.quad_list[0].position
        M = np.identity(2)
        for ind, quad in enumerate(self.quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = quad.length
            k = quad_strengths[ind]
            if k != 0:
                k_sqrt = np.sqrt(k*(1+0j))

                M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sin(k_sqrt * L) / k_sqrt],
                                        [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            else:
                M_q = np.array([[1, L], [0, 1]])
            M = np.matmul(M_q, M)
            s = quad.position + L
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

    def get_screen_beamsize(self, quad_strengths=None, axis="x"):
        if quad_strengths is None:
            M = self.calc_response_matrix(self.quad_strengths)
        else:
            if axis == "x":
                M = self.calc_response_matrix(quad_strengths)
            else:
                M = self.calc_response_matrix(-np.array(quad_strengths))
        a = M[0, 0]
        b = M[0, 1]
        sigma = np.sqrt(self.eps * (self.beta * a**2 - 2.0 * self.alpha * a * b + (1.0 + self.alpha**2) / self.beta * b**2))
        return sigma

    def target_beamsize(self, target_s):
        x0 = np.zeros(len(self.quad_strengths))

        def opt_fun(x, target_s):
            s = self.get_screen_beamsize(x)
            return (s-target_s)**2

        res = minimize(opt_fun, x0=x0, method="Nelder-Mead", args=(target_s, ), tol=target_s*0.1)
        return res


class MultiQuad(object):
    def __init__(self, alpha, beta, eps_n):
        self.name = "QuadSimulator"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))

        self.gamma_energy = 241e6 / 0.511e6
        self.max_quad_strength = 8.0    # T/m
        # self.max_k = 1.602e-19 * self.max_quad_strength / (self.gamma_energy * 9.11e-31 * 299792458.0)
        self.max_k = 5.0

        # self.sim = QuadSimulator(36.0, 69.0, 3e-6 * self.gamma_energy)  # MS-1 QB-01, SCRN-01
        self.sim = QuadSimulator(alpha, beta, eps_n / self.gamma_energy)  # MS-1 QB-01, SCRN-01
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
        # self.quad_strength_list = [1.0, -1.1, 1.9, 4.0]
        # self.quad_strength_list = [-7.14248646, -1.57912629,  4.05855788,  4.97507142]
        self.quad_strength_list = [2.0, -2.9, 1.7, -0.8]    # 1.0 mm size
        self.quad_strength_list = [-0.7, -0.3, -3.6, 2.3]    # 0.4 mm size

        self.screen = SectionScreen("screen", 19.223, "liveviewer", "beamviewer", "limaccd", "screen")

        self.algo = "const_size"

        self.a_list = list()
        self.b_list = list()
        self.target_a_list = list()
        self.target_b_list = list()
        self.M_list = list()
        self.k_list = list()
        self.x_list = list()
        self.y_list = list()
        self.psi_target = list()

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.eps_n_list = list()
        self.theta_list = list()
        self.r_maj_list = list()
        self.r_min_list = list()

        self.a_range = None
        self.b_range = None

        self.n_steps = 16

        self.logger.setLevel(logging.DEBUG)

    def scan(self, target_sigma, n_steps=16):
        M0 = self.calc_response_matrix(self.quad_strength_list, self.quad_list, self.screen.position)
        a0 = M0[0, 0]
        b0 = M0[0, 1]
        sigma0 = self.sim.get_screen_beamsize(self.quad_strength_list, axis="x")
        sigma_y0 = self.sim.get_screen_beamsize(self.quad_strength_list, axis="y")
        target_sigma = sigma0
        alpha0 = 0.0
        eps0 = 2e-6 / self.gamma_energy
        beta0 = self.get_missing_twiss(sigma0, M0, alpha0, None, eps0)
        if np.isnan(beta0):
            beta0 = 50.0
        # alpha0 = 36.0
        beta0 = 40.0
        theta, r_maj, r_min = self.calc_ellipse(alpha0, beta0, eps0, sigma0)
        # psi0 = np.arcsin(a0 / np.sqrt(a0**2 + b0**2)) - theta
        psi0 = np.arccos((a0 + b0 * np.tan(theta)) * np.cos(theta) / r_maj)

        a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
        self.a_range = (a_min, a_max)
        self.b_range = (b_min, b_max)

        self.logger.info("Scan starting. Initial parameters: th={0:.3f}, r_maj={1:.3f}, r_min={2:.3f}, "
                         "a0={3:.3f}, b0={4:.3f}, sigma0={5:.3f}, beta0={6:.3f}".format(theta, r_maj, r_min, a0, b0,
                                                                                        sigma0, beta0))

        self.n_steps = n_steps
        # self.psi_target = np.linspace(0, 1, n_steps) * 2 * np.pi + psi0
        # self.psi_target[1] = psi0 - 0.025
        # self.psi_target[2] = psi0 - 0.05
        self.psi_target = [psi0]
        self.k_list = [self.quad_strength_list]

        self.alpha_list.append(alpha0)
        self.beta_list.append(beta0)
        self.eps_list.append(eps0)
        self.eps_n_list.append(eps0 * self.gamma_energy)
        self.x_list.append(sigma0)
        self.y_list.append(sigma_y0)
        self.a_list.append(M0[0, 0])
        self.b_list.append(M0[0, 1])
        self.target_a_list.append(M0[0, 0])
        self.target_b_list.append(M0[0, 1])
        self.theta_list.append(theta)
        self.r_maj_list.append(r_maj)
        self.r_min_list.append(r_min)

        for step in range(self.n_steps - 1):
            # psi = self.psi_target[step + 1]
            a, b = self.set_target_ab(step, theta, r_maj, r_min)
            res = self.scan_step(a, b, target_sigma)
            if res is not None:
                self.logger.info("Step {0} SUCCEDED".format(step))
                theta = res[0]
                r_maj = res[1]
                r_min = res[2]
            else:
                self.logger.info("Step {0} FAILED".format(step))

    def scan_step(self, a, b, target_sigma):
        self.logger.info("{0} Scan step {1}, {2}".format(self, a, b))
        r = self.solve_quads(a, b)
        if r is None:
            return None
        M = self.calc_response_matrix(r.x, self.quad_list, self.screen.position)
        self.a_list.append(M[0, 0])
        self.b_list.append(M[0, 1])
        self.target_a_list.append(a)
        self.target_b_list.append(b)
        sigma = self.sim.get_screen_beamsize(r.x)
        self.x_list.append(sigma)
        sigma_y = self.sim.get_screen_beamsize(r.x, axis="y")
        self.y_list.append(sigma_y)
        res = self.calc_twiss(np.array(self.a_list), np.array(self.b_list), np.array(self.x_list))
        if res is not None:
            alpha = res[0]
            beta = res[1]
            eps = res[2]
        theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps, target_sigma)
        self.k_list.append(r.x)
        self.alpha_list.append(alpha)
        self.beta_list.append(beta)
        self.eps_list.append(eps)
        self.eps_n_list.append(eps * self.gamma_energy)
        self.theta_list.append(theta)
        self.r_maj_list.append(r_maj)
        self.r_min_list.append(r_min)
        return theta, r_maj, r_min

    def calc_twiss(self, a, b, sigma):
        M = np.vstack((a*a, -2*a*b, b*b)).transpose()
        # if M.shape[0] == 1:
        #     alpha = self.alpha_list[0]
        #     beta = self.beta_list[0]
        #     eps = sigma[0]**2 / (beta * M[0, 0] + alpha * M[0, 1] + (1 + alpha**2) / beta * M[0, 2])
        # elif M.shape[0] == 2:
        #     alpha = self.alpha_list[0]
        #     k = sigma[0]**2 / sigma[1]**2
        #     a1 = a[0]
        #     a2 = a[1]
        #     b1 = b[0]
        #     b2 = b[1]
        #     c1 = a1**2 - k * a2**2
        #     c2 = -2 * alpha * (a1 * b1 - k * a2 * b2)
        #     c3 = (1 + alpha**2) * (b1**2 - k * b2**2)
        #     beta = -c2 / (2 * c1) + np.sqrt(c2**2 / (4 * c1**2) - c3 / c1)
        #     eps = sigma[0]**2 / (beta * a1**2 - 2 * a1 * b1 * alpha + (1 + alpha**2) / beta * b1**2)
        if M.shape[0] < 3:
            # alpha0 = [self.alpha_list[0] - 10.0, self.alpha_list[0] + 10.0]
            # beta0 = [np.maximum(self.beta_list[0] - 10.0, 0), self.beta_list[0] + 10.0]
            # eps0 = [self.eps_list[0] * 0.8, self.eps_list[0] * 1.2]
            # bounds = ([beta0[0]*eps0[0], alpha0[0]*eps0[0], (1 + alpha0[1]**2) * eps0[0]**2 / (beta0[1] * eps0[1])],
            #           [beta0[1]*eps0[1], alpha0[1]*eps0[1], (1 + alpha0[1]**2) * eps0[1]**2 / (beta0[0] * eps0[0])])
            # l_data = lsq_linear(M, sigma**2, bounds=bounds)
            # x = l_data.x
            # self.logger.debug("Bounds: {0}\n result: {1}".format(bounds, l_data))
            # eps = np.sqrt(x[2] * x[0] - x[1]**2)
            # if np.isnan(eps):
            #     eps = self.eps_list[-1]
            # eps_n = eps / self.gamma_energy
            # beta = x[0] / eps
            # alpha = x[1] / eps
            alpha = self.alpha_list[-1]
            beta = self.beta_list[-1]
            eps = self.eps_list[-1]
        else:
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

    def set_target_ab(self, step, theta, r_maj, r_min):
        self.logger.debug("{0}: Determine new target a,b for algo {1}".format(self, self.algo))
        if self.algo == "const_size":
            if step < 2:
                psi = self.psi_target[-1] - 0.01
                target_a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
                target_b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
            else:
                a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
                psi_v = np.linspace(0, 2 * np.pi, 1000)
                a, b = self.get_ab(psi_v, theta, r_maj, r_min)
                ind = np.all([a < a_max, a > a_min, b < b_max, b > b_min], axis=0)
                a_g = a[ind]
                b_g = b[ind]
                st = int(a_g.shape[0] / self.n_steps + 0.5)
                target_a = a_g[::st][step]
                target_b = b_g[::st][step]
                psi = self.get_psi(target_a, target_b, theta, r_maj, r_min)

            self.psi_target.append(psi)
        else:
            target_a = 1
            target_b = 0
        # self.logger.debug("Target a, b = {0:.3f}, {1:.3f}".format(target_a, target_b))
        return target_a, target_b

    def solve_quads(self, target_a, target_b):
        self.logger.info("{0}: Solving new quad strengths for target a,b = {1}, {2}".format(self,
                                                                                                   target_a,
                                                                                                   target_b))
        x0 = self.quad_strength_list
        # x0 = self.k_list[-1]
        b = (-self.max_k, self.max_k)
        # res = minimize(self.opt_fun3, x0=x0, args=([target_a, target_b], self.alpha_list[-1], self.beta_list[-1],
        #                                            self.eps_list[-1], self.y_list[0]), bounds=(b, b, b, b))

        bfgs = BFGS()
        constraints = list()

        def constr_fun(x, target, a, u):
            M = self.calc_response_matrix(x, self.quad_list, self.screen.position)
            if a:
                comp = M[0, 0]
            else:
                comp = M[0, 1]
            if u:
                y = (target - 0.02) - comp
            else:
                y = comp - (target - 0.02)
            return y

        def constr_fun2(x):
            return -(np.abs(x) > 5.0).sum()

        constraints.append({"type": "ineq", "fun": constr_fun, "args": [target_a, True, True]})
        constraints.append({"type": "ineq", "fun": constr_fun, "args": [target_a, True, False]})
        constraints.append({"type": "ineq", "fun": constr_fun, "args": [target_b, False, True]})
        constraints.append({"type": "ineq", "fun": constr_fun, "args": [target_b, False, False]})
        constraints.append({"type": "ineq", "fun": constr_fun2})
        # res = minimize(self.opt_fun2, x0=x0, method="COBYLA", jac="2-point", hess=bfgs, constraints=constraints)
        # res = minimize(self.opt_fun4, x0=x0, method="COBYLA", jac="2-point", hess=bfgs, constraints=constraints,
        #                args=(self.alpha_list[-1], self.beta_list[-1],
        #                      self.eps_list[-1], self.x_list[0], self.y_list[0]))

        def tc_constr0(x):
            M = self.calc_response_matrix(x, self.quad_list, self.screen.position)
            y0 = (target_a - M[0, 0])**2
            y1 = (target_b - M[0, 1])**2
            return [y0, y1]

        def tc_constr1(x):
            M_x = self.calc_response_matrix(x, self.quad_list, self.screen.position, axis="x")
            a_x = M_x[0, 0]
            b_x = M_x[0, 1]
            M_y = self.calc_response_matrix(x, self.quad_list, self.screen.position, axis="y")
            a_y = M_y[0, 0]
            b_y = M_y[0, 1]
            alpha = self.alpha_list[-1]
            beta = self.beta_list[-1]
            eps = self.eps_list[-1]
            sigma_x = self.x_list[0]
            sigma_y = self.y_list[0]
            s_x = eps * (a_x ** 2 * beta - 2 * a_x * b_x * alpha + b_x ** 2 * (1 + alpha ** 2) / beta)
            s_y = eps * (a_y ** 2 * beta - 2 * a_y * b_y * alpha + b_y ** 2 * (1 + alpha ** 2) / beta)
            # y0 = (s_x / sigma_x - 1) ** 2
            # y1 = (s_y / sigma_y - 1) ** 2
            s_t = (s_x * s_y - sigma_x * sigma_y) ** 2 / (sigma_x * sigma_y) ** 2
            return [s_t]

        c0 = NonlinearConstraint(tc_constr0, 0, 0.2, jac="2-point", hess=bfgs)
        c1 = NonlinearConstraint(tc_constr1, 0, 0.02, jac="2-point", hess=bfgs)
        bounds = Bounds(-self.max_k, self.max_k)
        options = {"xtol": 1e-8, "verbose": 1}
        res = minimize(self.opt_fun4, x0=x0,  method="trust-constr", jac="2-point", hess=bfgs,
                       args=(self.alpha_list[-1], self.beta_list[-1], self.eps_list[-1], self.x_list[0], self.y_list[0]),
                       constraints=[c0], options=options, bounds=bounds)

        self.logger.debug("Found quad strengths: {0}".format(res.x))
        self.logger.debug("{0}".format(res))
        if res.status in [1, 2]:
            self.logger.debug("-------SUCCESS------")
        else:
            self.logger.debug("-----EPIC FAIL------")
        return res

    def opt_fun(self, x, target_ab):
        M = self.calc_response_matrix(x, self.quad_list, self.screen.position)
        y = (M[0, 0] - target_ab[0])**2 + (M[0, 1] - target_ab[1])**2
        w = np.sum(x**2) * 0.000
        return y + w

    def opt_fun3(self, x, target_ab, alpha, beta, eps, sigma_y):
        M_x = self.calc_response_matrix(x, self.quad_list, self.screen.position)
        M_y = self.calc_response_matrix(x, self.quad_list, self.screen.position, axis="y")
        s_x = (M_x[0, 0] - target_ab[0])**2 + (M_x[0, 1] - target_ab[1])**2
        a_y = M_y[0, 0]
        b_y = M_y[0, 1]
        s_y = (eps*(a_y**2 * beta - 2 * a_y * b_y * alpha + b_y**2 * (1 + alpha**2) / beta) - sigma_y)**2
        # self.logger.debug("s_x {0}, s_y {1}".format(s_x, s_y))
        w = np.sum(x**2) * 0.000
        w_y = 1e6
        return s_x + s_y * w_y

    def opt_fun4(self, x, alpha, beta, eps, sigma_x, sigma_y):
        M_x = self.calc_response_matrix(x, self.quad_list, self.screen.position)
        M_y = self.calc_response_matrix(x, self.quad_list, self.screen.position, axis="y")
        a_x = M_x[0, 0]
        b_x = M_x[0, 1]
        a_y = M_y[0, 0]
        b_y = M_y[0, 1]
        s_x = eps*(a_x**2 * beta - 2 * a_x * b_x * alpha + b_x**2 * (1 + alpha**2) / beta)
        s_y = eps * (a_y ** 2 * beta - 2 * a_y * b_y * alpha + b_y ** 2 * (1 + alpha ** 2) / beta)
        s_t = 1e6*(s_x / sigma_x - 1) ** 2 + 1e6*(s_y / sigma_y - 1) ** 2
        # s_t = (s_x * s_y - sigma_x * sigma_y)**2
        # self.logger.debug("s_x {0}, s_y {1}".format(s_x, s_y))
        return s_t

    def opt_fun2(self, x):
        w = np.sum(x**2)
        return w

    def calc_response_matrix(self, quad_strengths, quad_list, screen_position, axis="x"):
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = quad_list[0].position
        M = np.identity(2)
        if axis != "x":
            quad_strengths = -np.array(quad_strengths)
        for ind, quad in enumerate(quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = quad.length
            k = quad_strengths[ind]
            if k != 0:
                k_sqrt = np.sqrt(k*(1+0j))

                M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sin(k_sqrt * L) / k_sqrt],
                                        [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            else:
                M_q = np.array([[1, L], [0, 1]])
            M = np.matmul(M_q, M)
            s = quad.position + L
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
        if eps is None:
            res = sigma**2 / (beta * a**2 - 2.0 * alpha * a * b + (1.0 + alpha**2) / beta * b**2)
        elif beta is None:
            p = b * alpha / a + sigma**2 / (2 * eps * a**2)
            res = p - np.sqrt(p**2 - b**2 * (1 + alpha**2) / a**2)
        else:
            res = a * beta / b - np.sqrt(sigma**2 * beta / (b**2 * eps) - 1)
        return res

    def get_ab_range(self, max_k):
        """
        Calculate range of a and b that is reachable given a maximum magnet k-value
        :param max_k: Magnet maximum k-value k = (-max_k, max_k)
        :return: Minimum a, maxmium a, minimum b, maximum b
        """
        def ab_fun(x, a=True, maxmin=False):
            if a:
                if maxmin:
                    return -self.calc_response_matrix(x, self.quad_list, self.screen.position)[0, 0]
                else:
                    return self.calc_response_matrix(x, self.quad_list, self.screen.position)[0, 0]
            else:
                if maxmin:
                    return -self.calc_response_matrix(x, self.quad_list, self.screen.position)[0, 1]
                else:
                    return self.calc_response_matrix(x, self.quad_list, self.screen.position)[0, 1]
        c = 0.8
        mag_r = (-c * max_k, c * max_k)
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(True, False), bounds=(mag_r, mag_r, mag_r, mag_r))
        a_min = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(True, True), bounds=(mag_r, mag_r, mag_r, mag_r))
        a_max = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(False, False), bounds=(mag_r, mag_r, mag_r, mag_r))
        b_min = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 1]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(False, True), bounds=(mag_r, mag_r, mag_r, mag_r))
        b_max = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 1]
        return a_min, a_max, b_min, b_max

    def get_a(self, b, theta, r_maj, r_min):
        A = (r_maj * np.sin(theta))**2 + (r_min * np.cos(theta))**2
        B = - 2 * b * r_maj * np.sin(theta)
        C = b**2 - (r_min * np.cos(theta))**2
        cos_psi0 = (-B + np.sqrt(B**2 - 4 * C * A)) / (2 * A)
        cos_psi1 = (-B - np.sqrt(B ** 2 - 4 * C * A)) / (2 * A)
        a0 = r_maj * cos_psi0 / np.cos(theta) - b * np.tan(theta)
        a1 = r_maj * cos_psi1 / np.cos(theta) - b * np.tan(theta)
        return a0, a1

    def get_b(self, a, theta, r_maj, r_min):
        A = (r_maj * np.cos(theta))**2 + (r_min * np.sin(theta))**2
        B = - 2 * a * r_maj * np.cos(theta)
        C = a**2 - (r_min * np.sin(theta))**2
        cos_psi0 = (-B + np.sqrt(B**2 - 4 * C * A)) / (2 * A)
        cos_psi1 = (-B - np.sqrt(B ** 2 - 4 * C * A)) / (2 * A)
        b0 = r_maj * cos_psi0 / np.sin(theta) - a / np.tan(theta)
        b1 = r_maj * cos_psi1 / np.sin(theta) - a / np.tan(theta)
        return b0, b1

    def get_psi(self, a, b, theta, r_maj, r_min):
        p1 = (b * np.sin(theta) + a * np.cos(theta)) / r_maj
        p2 = (b * np.cos(theta) - a * np.sin(theta)) / r_min
        if abs(p1) < 0.5:
            psi = np.arccos(p1)
        else:
            psi = np.arcsin(p2)
        return psi

    def get_ab(self, psi, theta, r_maj, r_min):
        a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
        b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
        return a, b

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

    alpha = 35.0
    beta = 69.0
    eps = 1e-6
    sigma_target = 0.0005

    q_i = [1.87, -1.30, -0.24, 2.61]
    mq = MultiQuad(alpha, beta, eps)
    sigma_target = mq.sim.get_screen_beamsize(mq.quad_strength_list)
    # mq.quad_strength_list = q_i
    mq.scan(sigma_target)
    theta, r_maj, r_min = mq.calc_ellipse(alpha, beta, eps / mq.gamma_energy, sigma_target)
    psi_v = np.linspace(0, 2 * np.pi, 1000)
    a, b = mq.get_ab(psi_v, theta, r_maj, r_min)

