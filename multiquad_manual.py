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


try:
    import PyTango as pt
except ImportError:
    try:
        import tango as pt
    except ModuleNotFoundError:
        pass


class QuadSimulator(object):
    def __init__(self, alpha, beta, eps, alpha_y=None, beta_y=None, eps_y=None, add_noise=True):
        self.name = "QuadSimulator"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))
        self.logger.setLevel(logging.INFO)

        self.alpha = None
        self.beta = None
        self.eps = None
        self.sigma1 = None
        self.alpha_y = None
        self.beta_y = None
        self.eps_y = None
        self.sigma1_y = None

        self.add_noise = add_noise
        # Shot-to-shot errors
        self.noise_factors = {"alpha": 0.00, "beta": 0.000, "eps": 0.00, "sigma": 0.05, "quad": 0.00}
        # Systematic errors:
        self.pos_error = np.random.normal(0, 0.001, 6)
        self.quad_cal_error = np.random.normal(0, 0.001, 6)

        self.set_start_twiss_params(alpha, beta, eps, alpha_y, beta_y, eps_y)

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

    def set_start_twiss_params(self, alpha, beta, eps, alpha_y=None, beta_y=None, eps_y=None):
        self.logger.info("New Twiss parameters: alpha={0}, beta={1}, eps={2}".format(alpha, beta, eps))
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        if alpha_y is None:
            self.alpha_y = alpha
        else:
            self.alpha_y = alpha_y
        if beta_y is None:
            self.beta_y = beta
        else:
            self.beta_y = beta_y
        if eps_y is None:
            self.eps_y = eps
        else:
            self.eps_y = eps_y
        gamma = (1.0 + alpha**2) / beta
        self.sigma1 = eps * np.array([[beta, -alpha], [-alpha, gamma]])
        gamma_y = (1.0 + self.alpha_y**2) / self.beta_y
        self.sigma1_y = self.eps_y * np.array([[self.beta_y, -self.alpha_y], [-self.alpha_y, gamma_y]])

    def calc_response_matrix(self, quad_strengths, axis="x"):
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        if self.add_noise:
            s = self.quad_list[0].position + self.pos_error[0]
        else:
            s = self.quad_list[0].position
        M = np.identity(2)
        for ind, quad in enumerate(self.quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            if self.add_noise:
                drift = quad.position + self.pos_error[ind] - s
            else:
                drift = quad.position - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = quad.length
            if axis == "x":
                k = quad_strengths[ind]
            else:
                k = -quad_strengths[ind]
            if self.add_noise:
                k_sqrt = np.sqrt(k * (1 + self.quad_cal_error[ind] + 0j))
            else:
                k_sqrt = np.sqrt(k * (1 + 0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L), np.sinc(k_sqrt * (L / np.pi)) * L],
                                    [-k_sqrt * np.sin(k_sqrt * L), np.cos(k_sqrt * L)]]))
            # if k != 0:
            #     k_sqrt = np.sqrt(k*(1+0j))
            #
            #     M_q = np.real(np.array([[np.cos(k_sqrt * L),            np.sin(k_sqrt * L) / k_sqrt],
            #                             [-k_sqrt * np.sin(k_sqrt * L),  np.cos(k_sqrt * L)]]))
            # else:
            #     M_q = np.array([[1, L], [0, 1]])
            M = np.matmul(M_q, M)
            s = quad.position + L
        drift = self.screen.position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        return M

    def calc_response_matrix_m(self, quad_strengths, axis="x"):
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = self.quad_list[0].position
        d0 = list(quad_strengths[0].shape)
        d0.extend([1, 1])
        M = np.tile(np.identity(2), d0)
        M_d = M.copy()

        for ind, quad in enumerate(self.quad_list):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad.position - s
            M_d[:, :, :, :, 0, 1] = drift
            M = np.matmul(M_d, M)
            L = quad.length
            if axis == "x":
                k = quad_strengths[ind]
            else:
                k = -quad_strengths[ind]
            k_sqrt = np.sqrt(k * (1 + 0j))

            M_q0 = np.real(np.array([[np.cos(k_sqrt * L), np.sinc(k_sqrt * (L / np.pi)) * L],
                                    [-k_sqrt * np.sin(k_sqrt * L), np.cos(k_sqrt * L)]]))
            M_q = np.moveaxis(M_q0, [0, 1], [-2, -1])
            M = np.matmul(M_q, M)
            s = quad.position + L
        drift = self.screen.position - s
        M_d[:, :, :, :, 0, 1] = drift
        M = np.matmul(M_d, M)
        return M

    def get_screen_twiss(self, axis="x"):
        M = self.calc_response_matrix(self.quad_strengths, axis)
        if axis == "x":
            sigma2 = np.matmul(np.matmul(M, self.sigma1), M.transpose())
        else:
            sigma2 = np.matmul(np.matmul(M, self.sigma1_y), M.transpose())
        eps = np.sqrt(np.linalg.det(sigma2))
        alpha = -sigma2[0, 1] / eps
        beta = sigma2[0, 0] / eps
        return alpha, beta, eps

    def get_screen_beamsize(self, quad_strengths=None, axis="x"):
        if quad_strengths is None:
            q = np.array(self.quad_strengths)
        else:
            q = np.array(quad_strengths)
        if self.add_noise:
            for k in range(len(q)):
                q[k] *= (1 + np.random.normal(0, self.noise_factors["quad"]))
        M = self.calc_response_matrix(q, axis)
        a = M[0, 0]
        b = M[0, 1]
        if axis == "x":
            if self.add_noise:
                alpha = self.alpha * (1 + np.random.normal(0, self.noise_factors["alpha"]))
                beta = self.beta * (1 + np.random.normal(0, self.noise_factors["beta"]))
                eps = self.eps * (1 + np.random.normal(0, self.noise_factors["eps"]))
                self.logger.debug("Noisy twiss: {0}, {1}, {2}".format(alpha, beta, eps))
                sigma = np.sqrt(eps * (beta * a**2 - 2.0 * alpha * a * b + (1.0 + alpha**2) / beta * b**2)) * \
                        (1 + self.noise_factors["sigma"])
            else:
                sigma = np.sqrt(self.eps * (
                        self.beta * a**2 - 2.0 * self.alpha * a * b + (1.0 + self.alpha**2) / self.beta * b**2))
        else:
            if self.add_noise:
                alpha = self.alpha_y * (1 + np.random.normal(0, self.noise_factors["alpha"]))
                beta = self.beta_y * (1 + np.random.normal(0, self.noise_factors["beta"]))
                eps = self.eps_y * (1 + np.random.normal(0, self.noise_factors["eps"]))
                sigma = np.sqrt(eps * (beta * a ** 2 - 2.0 * alpha * a * b + (1.0 + alpha ** 2) / beta * b ** 2)) * \
                        (1 + self.noise_factors["sigma"])
            else:
                sigma = np.sqrt(self.eps_y * (
                        self.beta_y * a ** 2 - 2.0 * self.alpha_y * a * b + (1.0 + self.alpha_y ** 2) / self.beta_y * b ** 2))
        return sigma

    def target_beamsize(self, target_s):
        x0 = np.zeros(len(self.quad_strengths))

        def opt_fun(x, target_s):
            s = self.get_screen_beamsize(x)
            return (s-target_s)**2

        res = minimize(opt_fun, x0=x0, method="Nelder-Mead", args=(target_s, ), tol=target_s*0.1)
        return res


class MultiQuadManual(object):
    def __init__(self):
        self.name = "MultiQuadScan"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))

        self.max_k = 5.0

        self.algo = "const_size"
        self.gamma_energy = None
        self.quad_list = None
        self.quad_strength_list = None
        self.screen = None

        self.a_list = list()
        self.b_list = list()
        self.a_y_list = list()
        self.b_y_list = list()
        self.target_a_list = list()
        self.target_b_list = list()
        self.M_list = list()
        self.k_list = list()
        self.x_list = list()
        self.y_list = list()
        self.psi_target = list()
        self.psi_list = list()

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.eps_n_list = list()
        self.alpha_y_list = list()
        self.beta_y_list = list()
        self.eps_y_list = list()
        self.eps_n_y_list = list()
        self.theta_list = list()
        self.r_maj_list = list()
        self.r_min_list = list()

        self.a_range = None
        self.b_range = None

        self.n_steps = 16
        self.current_step = 0
        self.target_sigma = None

        self.guess_alpha = None
        self.guess_beta = None
        self.guess_eps_n = None

        self.logger.setLevel(logging.DEBUG)

    def reset_data(self):
        self.a_list = list()
        self.b_list = list()
        self.a_y_list = list()
        self.b_y_list = list()
        self.target_a_list = list()
        self.target_b_list = list()
        self.M_list = list()
        self.k_list = list()
        self.x_list = list()
        self.y_list = list()
        self.psi_target = list()
        self.psi_list = list()

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.eps_n_list = list()
        self.alpha_y_list = list()
        self.beta_y_list = list()
        self.eps_y_list = list()
        self.eps_n_y_list = list()
        self.theta_list = list()
        self.r_maj_list = list()
        self.r_min_list = list()

        self.a_range = None
        self.b_range = None

        self.n_steps = 16
        self.current_step = 0
        self.target_sigma = None

        self.guess_alpha = None
        self.guess_beta = None
        self.guess_eps_n = None

    def start_scan(self, current_sigma_x, section="MS1", n_steps=16, guess_alpha=0.0, guess_beta=40.0, guess_eps_n=2e-6):
        """
        Start a new multi-quad scan using current beam size as target beam size. Initial guess should be provided
        if known.

        :param current_sigma_x: Current horizontal beam size sigma, used as target beam size for scan
        :param section: MS1, MS2, MS3
        :param n_steps: Number of steps in scan. Determines spacing of a-b values on the ellipse
        :param guess_alpha: Staring guess of beam twiss parameter alpha. Used for first two steps.
        :param guess_beta: Staring guess of beam twiss parameter beta. Used for first two steps.
        :param guess_eps_n: Staring guess of beam twiss parameter normalized emittance. Used for first two steps.
        :return:
        """
        self.reset_data()
        self.set_section(section)
        self.target_sigma = current_sigma_x

        self.guess_alpha = guess_alpha
        self.guess_beta = guess_beta
        self.guess_eps_n = guess_eps_n

        a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
        self.a_range = (a_min, a_max)
        self.b_range = (b_min, b_max)

        self.logger.info("Scan starting for section {0}. Target sigma: {1:.3f} mm".format(section, 1e3 * current_sigma_x))

        self.n_steps = n_steps
        self.current_step = 0

    def scan_step(self, current_sigma_x, current_sigma_y, current_k_list):
        """
        Do a scan step with manual input of beam size sigma and quad magnet k values

        :param current_sigma_x: Horizontal beam size sigma for current setting (m)
        :param current_sigma_y: Vertical beam size sigma for current setting (m)
        :param current_k_list: List of k-values for the section (MS1, MS2: for quads, MS3: 6 quads)
        :return: k-values for next point
        """
        self.current_step += 1
        self.k_list.append(current_k_list)

        # Calculate response matrix for current quad settings and store the a-b values
        M = self.calc_response_matrix(current_k_list, self.quad_list, self.screen.position, axis="x")
        self.a_list.append(M[0, 0])
        self.b_list.append(M[0, 1])

        # Calculate new estimates of twiss parameters including the current point
        self.x_list.append(current_sigma_x)
        res = self.calc_twiss(np.array(self.a_list), np.array(self.b_list), np.array(self.x_list))
        if res is not None:
            alpha = res[0]
            beta = res[1]
            eps = res[2]
        self.alpha_list.append(alpha)
        self.beta_list.append(beta)
        self.eps_list.append(eps)
        self.eps_n_list.append(eps * self.gamma_energy)

        M_y = self.calc_response_matrix(current_k_list, self.quad_list, self.screen.position, axis="y")
        self.a_y_list.append(M_y[0, 0])
        self.b_y_list.append(M_y[0, 1])
        self.y_list.append(current_sigma_y)

        res_y = self.calc_twiss(np.array(self.a_y_list), np.array(self.b_y_list), np.array(self.y_list))
        if res_y is not None:
            alpha_y = res_y[0]
            beta_y = res_y[1]
            eps_y = res_y[2]

        # Calculate new estimate of ellipse from the updated twiss parameters
        theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps, self.target_sigma)
        if np.isnan(theta):
            theta = self.theta_list[-1]
        if np.isnan(r_maj):
            r_maj = self.r_maj_list[-1]
        if np.isnan(r_min):
            r_min = self.r_min_list[-1]

        # Calculate next a-b values to target
        next_a, next_b = self.set_target_ab(self.current_step, theta, r_maj, r_min)

        # Determine quad settings to achieve these a-b values
        r = self.solve_quads(next_a, next_b)
        if r is None:
            self.logger.error("Could not find quad settings to match desired a-b values")
            self.k_list.pop()
            self.x_list.pop()
            self.a_list.pop()
            self.b_list.pop()
            self.y_list.pop()
            self.a_y_list.pop()
            self.b_y_list.pop()
            self.alpha_list.pop()
            self.beta_list.pop()
            self.eps_list.pop()
            self.eps_n_list.pop()
            return None
        next_k = r.x

        self.target_a_list.append(next_a)
        self.target_b_list.append(next_b)
        self.psi_list.append(self.get_psi(M[0, 0], M[0, 1], theta, r_maj, r_min))
        self.theta_list.append(theta)
        self.r_maj_list.append(r_maj)
        self.r_min_list.append(r_min)
        self.alpha_y_list.append(alpha_y)
        self.beta_y_list.append(beta_y)
        self.eps_y_list.append(eps_y)
        self.eps_n_y_list.append(eps_y * self.gamma_energy)

        s = "\n=================================\n" \
            "STEP {0}/{1} result:\n\n" \
            "alpha = {2:.3f}\n" \
            "beta  = {3:.3f}\n" \
            "eps_n = {4:.3g}\n\n" \
            "Next step magnet settings:" \
            "{5}".format(self.current_step, self.n_steps, alpha, beta, eps * self.gamma_energy, next_k)
        self.logger.info(s)

        return next_k

    def calc_twiss(self, a, b, sigma):
        M = np.vstack((a*a, -2*a*b, b*b)).transpose()
        if M.shape[0] < 3:
            alpha = self.guess_alpha
            beta = self.guess_beta
            eps = self.guess_eps_n / self.gamma_energy
        else:
            def opt_fun(x, a, b, sigma):
                return 1e6 * x[0] * (x[1] * a**2 - 2 * x[2] * a * b + (1 + x[2]**2) / x[1] * b**2) - (sigma * 1e3)**2

            x0 = [self.eps_list[0], self.beta_list[0], self.alpha_list[0]]
#            ldata = least_squares(opt_fun, x0, jac="2-point", args=(a, b, sigma),
#                                  bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), gtol=1e-16, xtol=1e-16, ftol=1e-16)
            ldata = least_squares(opt_fun, x0, jac="2-point", args=(a, b, sigma),
                                  bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
            eps = ldata.x[0]
            beta = ldata.x[1]
            alpha = ldata.x[2]
        return alpha, beta, eps

    def set_target_ab(self, step, theta, r_maj, r_min):
        self.logger.debug("{0}: Determine new target a,b for algo {1}".format(self, self.algo))
        if self.algo == "const_size":
            if step < 3:

                try:
                    psi = self.psi_target[-1] - 0.01
                except IndexError:
                    psi = np.arccos((self.a_list[0] + self.b_list[0] * np.tan(theta)) * np.cos(theta) / r_maj)
                target_a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
                target_b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
            else:
                a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
                psi_v = np.linspace(0, 2 * np.pi, 5000)
                a, b = self.get_ab(psi_v, theta, r_maj, r_min)
                ind = np.all([a < a_max, a > a_min, b < b_max, b > b_min], axis=0)
                a_g = a[ind]
                b_g = b[ind]
                st = int(a_g.shape[0] / self.n_steps + 0.5)
                self.logger.debug("Found {0} values in range. Using every {1} value".format(a_g.shape[0], st))
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
        self.logger.info("{0}: Solving new quad strengths for target a,b = {1:.3f}, {2:.3f}".format(self, target_a, target_b))
        # x0 = self.quad_strength_list
        x0 = self.k_list[-1]

        bfgs = BFGS()

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
            s_x = np.sqrt(eps * (a_x ** 2 * beta - 2 * a_x * b_x * alpha + b_x ** 2 * (1 + alpha ** 2) / beta))
            s_y = np.sqrt(eps * (a_y ** 2 * beta - 2 * a_y * b_y * alpha + b_y ** 2 * (1 + alpha ** 2) / beta))
            # y0 = (s_x / sigma_x - 1) ** 2
            # y1 = (s_y / sigma_y - 1) ** 2
            s_t = (s_x * s_y - sigma_x * sigma_y) ** 2 / (sigma_x * sigma_y) ** 2
            # s_t = (s_y - sigma_y) ** 2 / (sigma_y) ** 2
            return [s_t]

        def tc_constr2(x):
            M_x = self.calc_response_matrix(x, self.quad_list, self.screen.position)
            a_x = M_x[0, 0]
            b_x = M_x[0, 1]
            psi_x = self.get_psi(a_x, b_x, self.theta_list[-1], self.r_maj_list[-1], self.r_min_list[-1])
            s_t = (target_psi - psi_x) ** 2
            return [s_t]

        def opt_fun(x, alpha, beta, eps, sigma_x, sigma_y):
            M_x = self.calc_response_matrix(x, self.quad_list, self.screen.position)
            M_y = self.calc_response_matrix(x, self.quad_list, self.screen.position, axis="y")
            a_x = M_x[0, 0]
            b_x = M_x[0, 1]
            a_y = M_y[0, 0]
            b_y = M_y[0, 1]
            s_x = (eps * (a_x ** 2 * beta - 2 * a_x * b_x * alpha + b_x ** 2 * (1 + alpha ** 2) / beta))
            s_y = (eps * (a_y ** 2 * beta - 2 * a_y * b_y * alpha + b_y ** 2 * (1 + alpha ** 2) / beta))
            s_t = (s_x * s_y - sigma_x ** 2 * sigma_y ** 2) ** 2
            return s_t

        c0 = NonlinearConstraint(tc_constr0, 0, 0.1, jac="2-point", hess=bfgs)
        c1 = NonlinearConstraint(tc_constr1, 0, 0.02, jac="2-point", hess=bfgs)
        c2 = NonlinearConstraint(tc_constr2, 0, 0.0001, jac="2-point", hess=bfgs)
        bounds = Bounds(-self.max_k, self.max_k)
        options = {"xtol": 1e-8, "verbose": 1, "initial_constr_penalty": 100}
        res = minimize(opt_fun, x0=x0,  method="trust-constr", jac="2-point", hess=bfgs,
                       args=(self.alpha_list[-1], self.beta_list[-1], self.eps_list[-1], self.x_list[0], self.y_list[0]),
                       constraints=[c0], options=options, bounds=bounds)

        self.logger.debug("Found quad strengths: {0}".format(res.x))
        # self.logger.debug("{0}".format(res))
        if res.status in [1, 2]:
            self.logger.debug("-------SUCCESS------")
        else:
            self.logger.debug("-----EPIC FAIL------")
        return res

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
        self.logger.debug("Twiss indata: alpha={0:.3f}, beta={1:.3f}, eps={2:.3g}, sigma={3:.3g}".format(alpha, beta, eps, sigma))
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
        self.logger.debug("Result: theta={0:.3f}, r_maj={1:.3f}, r_min={2:.3f}".format(theta, r_major, r_minor))
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
        mag_r = Bounds(-c * max_k, c * max_k)
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(True, False), bounds=mag_r)
        a_min = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(True, True), bounds=mag_r)
        a_max = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(False, False), bounds=mag_r)
        b_min = mq.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 1]
        res = minimize(ab_fun, x0=self.quad_strength_list, args=(False, True), bounds=mag_r)
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
        psi = np.atleast_1d(np.arccos(p1))
        ind_q3 = np.logical_and(p1 < 0, p2 < 0)
        psi[ind_q3] = 2 * np.pi - psi[ind_q3]
        ind_q4 = np.logical_and(p1 > 0, p2 < 0)
        psi[ind_q4] = 2 * np.pi - psi[ind_q4]
        if psi.shape[0] == 1:
            psi = psi[0]
        return psi

    def get_ab(self, psi, theta, r_maj, r_min):
        a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
        b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
        return a, b

    def set_section(self, section):
        if section == "MS1":
            self.gamma_energy = 241e6 / 0.511e6
            self.quad_list = list()
            self.quad_list.append(SectionQuad("QB-01", 13.55, 0.2, "MAG-01", "CRQ-01", True))
            self.quad_list.append(SectionQuad("QB-02", 14.45, 0.2, "MAG-02", "CRQ-02", True))
            self.quad_list.append(SectionQuad("QB-03", 17.75, 0.2, "MAG-03", "CRQ-03", True))
            self.quad_list.append(SectionQuad("QB-04", 18.65, 0.2, "MAG-04", "CRQ-04", True))

            self.screen = SectionScreen("screen", 19.223, "liveviewer", "beamviewer", "limaccd", "screen")

            self.quad_strength_list = [-0.7, -0.3, -3.6, 2.3]  # 0.4 mm size

        elif section == "MS2":
            self.gamma_energy = 241e6 / 0.511e6

            self.quad_list = list()
            self.quad_list.append(SectionQuad("QB-01", 33.52, 0.2, "MAG-01", "CRQ-01", True))
            self.quad_list.append(SectionQuad("QB-02", 34.62, 0.2, "MAG-02", "CRQ-02", True))
            self.quad_list.append(SectionQuad("QB-03", 35.62, 0.2, "MAG-03", "CRQ-03", True))
            self.quad_list.append(SectionQuad("QB-04", 37.02, 0.2, "MAG-04", "CRQ-04", True))
            self.screen = SectionScreen("screen", 38.445, "liveviewer", "beamviewer", "limaccd", "screen")

            self.quad_strength_list = [-0.7, -0.3, -3.6, 2.3]  # 0.4 mm size

        elif section == "MS3":
            self.gamma_energy = 3020e6 / 0.511e6

            self.quad_list = list()
            self.quad_list.append(SectionQuad("QF-01", 275.719, 0.2, "MAG-01", "CRQ-01", True))
            self.quad_list.append(SectionQuad("QF-02", 277.719, 0.2, "MAG-02", "CRQ-02", True))
            self.quad_list.append(SectionQuad("QF-03", 278.919, 0.2, "MAG-03", "CRQ-03", True))
            self.quad_list.append(SectionQuad("QF-04", 281.119, 0.2, "MAG-04", "CRQ-04", True))
            self.quad_list.append(SectionQuad("QF-03", 281.619, 0.2, "MAG-03", "CRQ-05", True))
            self.quad_list.append(SectionQuad("QF-04", 282.019, 0.2, "MAG-04", "CRQ-06", True))
            self.screen = SectionScreen("screen", 282.456, "liveviewer", "beamviewer", "limaccd", "screen")

            self.quad_strength_list = [-0.7, -0.3, -3.6, 2.3, 1.0, 1.0]

    def set_k_values(self, k_list):
        self.k_list.append(k_list)

    def set_number_steps(self, n_steps):
        self.n_steps

    def __repr__(self):
        s = "{0} {1}".format(type(self), self.name)
        return s

    def __str__(self):
        s = "{0} {1}".format(type(self).__name__, self.name)
        return s


if __name__ == "__main__":

    mq = MultiQuadManual()
    sigma_target = 400e-6
    section = "MS1"
    n_steps = 8
    alpha0 = 0
    beta0 = 40
    eps_n_0 = 2e-6
    mq.start_scan(sigma_target, section, n_steps, alpha0, beta0, eps_n_0)
    mq.scan_step(400e-6, 340e-6, [-0.7, -0.3, -3.6, 2.3])
