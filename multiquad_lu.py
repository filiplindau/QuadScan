"""
Created 2020-03-13

Using a look-up table to find correct quad settings

@author: Filip Lindau
"""

import time
import ctypes
import inspect
import multiprocessing as mp
# from queue import Empty
# import PIL
import numpy as np
from numpngw import write_png
import os
from collections import namedtuple
import pprint
import traceback
from scipy.signal import medfilt2d
from scipy.optimize import minimize, Bounds, least_squares #, BFGS, NonlinearConstraint
from scipy.optimize import lsq_linear
# from QuadScanTasks import TangoReadAttributeTask, TangoMonitorAttributeTask, TangoWriteAttributeTask, work_func_local2
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


def sinc(x):
    y = np.sin(x) / x
    y[np.isnan(y)] = 1.0
    return y


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
        self.logger.info("Beam size {0}: {1:.3f} mm".format(axis, sigma*1e3))
        return sigma

    def target_beamsize(self, target_s):
        x0 = np.zeros(len(self.quad_strengths))

        def opt_fun(x, target_s):
            s = self.get_screen_beamsize(x)
            return (s-target_s)**2

        res = minimize(opt_fun, x0=x0, method="Nelder-Mead", args=(target_s, ), tol=target_s*0.1)
        return res


def calc_response_matrix_mp(quad_strengths, quad_positions, screen_position, axis="x"):
    # self.logger.debug("{0}: Calculating new response matrix".format(self))
    s = quad_positions[0]
    M = np.identity(2)
    if axis != "x":
        quad_strengths = -np.array(quad_strengths)
    for ind, quad in enumerate(quad_positions):
        # self.logger.debug("Position s: {0} m".format(s))
        drift = quad - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        L = 0.2
        k = quad_strengths[..., ind]
        k_sqrt = np.sqrt(k * (1 + 0j))

        M_q = np.real(np.array([[np.cos(k_sqrt * L), L * sinc(L * k_sqrt)],
                                [-k_sqrt * np.sin(k_sqrt * L), np.cos(k_sqrt * L)]]))
        M = np.matmul(np.moveaxis(M_q, (0, 1), (-2, -1)), M)
        s = quad + L
    drift = screen_position - s
    M_d = np.array([[1.0, drift], [0.0, 1.0]])
    M = np.matmul(M_d, M)
    return M


class MultiQuadLookup(object):
    def __init__(self):
        self.name = "MultiQuadScan"
        self.logger = logging.getLogger("Sim.{0}".format(self.name.upper()))

        self.max_k = 8.0

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
        self.target_charge = None
        self.charge_list = list()

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.eps_n_list = list()
        self.alpha_y_list = list()
        self.beta_y_list = list()
        self.eps_y_list = list()
        self.eps_n_y_list = list()
        self.theta_list = list()
        self.theta_y_list = list()
        self.r_maj_list = list()
        self.r_maj_y_list = list()
        self.r_min_list = list()
        self.r_min_y_list = list()

        self.a_range = None
        self.b_range = None

        self.A_lu = None
        self.k_lu = None

        self.n_steps = 16
        self.current_step = 0
        self.target_sigma = None
        self.target_sigma_y = None

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
        self.charge_list = list()
        self.target_charge = None

        self.alpha_list = list()
        self.beta_list = list()
        self.eps_list = list()
        self.eps_n_list = list()
        self.alpha_y_list = list()
        self.beta_y_list = list()
        self.eps_y_list = list()
        self.eps_n_y_list = list()
        self.theta_list = list()
        self.theta_y_list = list()
        self.r_maj_list = list()
        self.r_maj_y_list = list()
        self.r_min_list = list()
        self.r_min_y_list = list()

        self.A_lu = None
        self.k_lu = None

        self.a_range = None
        self.b_range = None

        self.n_steps = 16
        self.current_step = 0
        self.target_sigma = None

        self.guess_alpha = None
        self.guess_beta = None
        self.guess_eps_n = None

    def generate_lookup(self):
        self.logger.info("Generating lookup table for {0}".format(self.screen.screen))
        quad_positions = np.array([q.position for q in self.quad_list])
        screen_position = self.screen.position
        q_v = np.linspace(-self.max_k, self.max_k, 40)
        q1, q2, q3, q4 = np.meshgrid(q_v, q_v, q_v, q_v)
        quad_strengths = np.stack((q1, q2, q3, q4), -1)
        t0 = time.time()
        M = self.calc_response_matrix_v(quad_strengths, quad_positions, screen_position, "x")
        ax = M[..., 0, 0]
        bx = M[..., 0, 1]
        self.logger.debug("Time x: {0}".format(time.time() - t0))
        t1 = time.time()
        M = self.calc_response_matrix_v(quad_strengths, quad_positions, screen_position, "y")
        ay = M[..., 0, 0]
        by = M[..., 0, 1]
        self.logger.debug("Time y: {0}".format(time.time() - t1))
        A = np.stack((ax, bx, ay, by), -1).reshape(-1, 4)
        k = quad_strengths.reshape(-1, 4)
        self.A_lu = A
        self.k_lu = k

    def generate_lookup_mp(self):
        self.logger.info("Generating lookup table for {0}".format(self.screen.screen))
        quad_positions = np.array([q.position for q in self.quad_list])
        screen_position = self.screen.position
        q_v = np.linspace(-self.max_k, self.max_k, 40)
        q1, q2, q3, q4 = np.meshgrid(q_v, q_v, q_v, q_v)
        quad_strengths = np.stack((q1, q2, q3, q4), -1)
        t0 = time.time()
        n_proc = mp.cpu_count()

        with mp.Pool(processes=n_proc) as pool:
            quad_strengths_list = zip(np.array_split(quad_strengths, n_proc, 0),
                                      [quad_positions] * n_proc, [screen_position] * n_proc, ["x"] * n_proc)
            M = np.array(pool.starmap(calc_response_matrix_mp, quad_strengths_list))
            ax = M[..., 0, 0]
            bx = M[..., 0, 1]
            self.logger.debug("Time x: {0:.3f} s".format(time.time() - t0))
            t1 = time.time()
            quad_strengths_list = zip(np.array_split(quad_strengths, n_proc, 0),
                                      [quad_positions]*n_proc, [screen_position]*n_proc, ["y"]*n_proc)
            M = np.array(pool.starmap(calc_response_matrix_mp, quad_strengths_list))
            ay = M[..., 0, 0]
            by = M[..., 0, 1]
            self.logger.debug("Time y: {0:.3f} s".format(time.time() - t1))
        A = np.stack((ax, bx, ay, by), -1).reshape(-1, 4)
        k = quad_strengths.reshape(-1, 4)
        self.A_lu = A
        self.k_lu = k

    def start_scan(self, current_sigma_x, current_sigma_y, current_charge, section="MS1", n_steps=16,
                   guess_alpha=0.0, guess_beta=40.0, guess_eps_n=2e-6):
        """
        Start a new multi-quad scan using current beam size as target beam size. Initial guess should be provided
        if known.

        :param current_sigma_x: Current horizontal beam size sigma, used as target beam size for scan
        :param current_sigma_y: Current vertical beam size sigma, used as target beam size for scan
        :param current_charge: Total charge in image. Used to adjust the weight of different shots when calculating
        beam parameters
        :param section: MS1, MS2, MS3
        :param n_steps: Number of steps in scan. Determines spacing of a-b values on the ellipse
        :param guess_alpha: Staring guess of beam twiss parameter alpha. Used for first two steps.
        :param guess_beta: Staring guess of beam twiss parameter beta. Used for first two steps.
        :param guess_eps_n: Staring guess of beam twiss parameter normalized emittance. Used for first two steps.
        :return:
        """
        self.logger.info("Start scan: sigma {0:.3f} x {1:.3f} mm".format(current_sigma_x*1e3, current_sigma_y*1e3))
        self.reset_data()
        self.set_section(section)
        self.generate_lookup_mp()
        self.target_sigma = current_sigma_x
        self.target_sigma_y = current_sigma_y
        self.target_charge = current_charge

        self.guess_alpha = guess_alpha
        self.guess_beta = guess_beta
        self.guess_eps_n = guess_eps_n

        a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
        self.a_range = (a_min, a_max)
        self.b_range = (b_min, b_max)

        self.logger.info("Scan starting for section {0}. Target sigma: {1:.3f} mm".format(section, 1e3 * current_sigma_x))

        self.n_steps = n_steps
        self.current_step = 0

    def scan_step(self, current_sigma_x, current_sigma_y, current_charge, current_k_list):
        """
        Do a scan step with manual input of beam size sigma and quad magnet k values

        :param current_sigma_x: Horizontal beam size sigma for current setting (m)
        :param current_sigma_y: Vertical beam size sigma for current setting (m)
        :param current_k_list: List of k-values for the section (MS1, MS2: for quads, MS3: 6 quads)
        :return: k-values for next point
        """
        self.current_step += 1
        self.k_list.append(current_k_list)
        self.charge_list.append(current_charge)

        # Calculate response matrix for current quad settings and store the a-b values
        M = self.calc_response_matrix(current_k_list, self.quad_list, self.screen.position, axis="x")
        self.a_list.append(M[0, 0])
        self.b_list.append(M[0, 1])

        # Calculate new estimates of twiss parameters including the current point
        self.x_list.append(current_sigma_x)
        res = self.calc_twiss(np.array(self.a_list), np.array(self.b_list), np.array(self.x_list),
                              np.array(self.charge_list), "x")
        if res is not None:
            alpha = res[0]
            beta = res[1]
            eps = res[2]
        self.alpha_list.append(alpha)
        self.beta_list.append(beta)
        self.eps_list.append(eps)
        self.eps_n_list.append(eps * self.gamma_energy)

        # Calculate new estimate of ellipse from the updated twiss parameters
        theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps, self.target_sigma)
        if np.isnan(theta):
            theta = self.theta_list[-1]
        if np.isnan(r_maj):
            r_maj = self.r_maj_list[-1]
        if np.isnan(r_min):
            r_min = self.r_min_list[-1]

        M_y = self.calc_response_matrix(current_k_list, self.quad_list, self.screen.position, axis="y")
        self.a_y_list.append(M_y[0, 0])
        self.b_y_list.append(M_y[0, 1])
        self.y_list.append(current_sigma_y)

        res_y = self.calc_twiss(np.array(self.a_y_list), np.array(self.b_y_list), np.array(self.y_list),
                                np.array(self.charge_list), "y")
        if res_y is not None:
            alpha_y = res_y[0]
            beta_y = res_y[1]
            eps_y = res_y[2]
        self.alpha_y_list.append(alpha_y)
        self.beta_y_list.append(beta_y)
        self.eps_y_list.append(eps_y)
        self.eps_n_y_list.append(eps_y * self.gamma_energy)

        # Calculate new estimate of ellipse from the updated twiss parameters
        theta_y, r_maj_y, r_min_y = self.calc_ellipse(alpha_y, beta_y, eps_y, self.target_sigma_y)
        if np.isnan(theta_y):
            theta_y = self.theta_y_list[-1]
        if np.isnan(r_maj_y):
            r_maj_y = self.r_maj_y_list[-1]
        if np.isnan(r_min_y):
            r_min_y = self.r_min_y_list[-1]

        # Calculate next a-b values to target
        next_a, next_b = self.set_target_ab(self.current_step, theta, r_maj, r_min, axis="x")
        next_a_y, next_b_y = self.set_target_ab(self.current_step, theta_y, r_maj_y, r_min_y, axis="y")

        # Determine quad settings to achieve these a-b values
        next_k = self.solve_quads(next_a, next_b, next_a_y, next_b_y)
        if next_k is None:
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
            self.alpha_y_list.pop()
            self.beta_y_list.pop()
            self.eps_y_list.pop()
            self.eps_n_y_list.pop()
            self.charge_list.pop()
            return None

        self.target_a_list.append(next_a)
        self.target_b_list.append(next_b)
        self.psi_list.append(self.get_psi(M[0, 0], M[0, 1], theta, r_maj, r_min))
        self.theta_list.append(theta)
        self.r_maj_list.append(r_maj)
        self.r_min_list.append(r_min)
        self.theta_y_list.append(theta_y)
        self.r_maj_y_list.append(r_maj_y)
        self.r_min_y_list.append(r_min_y)

        s = "\n=================================\n" \
            "STEP {0}/{1} result:\n\n" \
            "alpha = {2:.3f}\n" \
            "beta  = {3:.3f}\n" \
            "eps_n = {4:.3g}\n\n" \
            "Next step magnet settings:\n" \
            "{5}".format(self.current_step, self.n_steps, alpha, beta, eps * self.gamma_energy, next_k)
        self.logger.info(s)
        if self.current_step >= self.n_steps:
            self.logger.info("=========== END OF SCAN ===========")

        return next_k

    def calc_twiss(self, a, b, sigma, charge=None, axis="x"):
        M = np.vstack((a*a, -2*a*b, b*b)).transpose()
        if M.shape[0] == 1:
            eps0 = self.guess_eps_n / self.gamma_energy
            alpha0 = self.guess_alpha
            beta0 = self.guess_beta
            c = np.sqrt(np.abs((sigma[-1]**2 / eps0 - b[-1]**2 / beta0) / (a[-1]**2 * beta0 - 2 * a[-1] * b[-1] * alpha0 +
                                                                    b[-1]**2 * alpha0**2 / beta0)))
            alpha = c * alpha0
            beta = c * beta0
            eps = c * eps0
        elif M.shape[0] == 2:
            theta = np.arctan(b[-1] / a[-1])
            if axis == "x":
                alpha = self.alpha_list[-1]
                beta = self.beta_list[-1]
            else:
                alpha = self.alpha_y_list[-1]
                beta = self.beta_y_list[-1]
            eps = sigma[-1]**2 / (a[-1]**2 * beta - 2 * a[-1] * b[-1] * alpha + b[-1]**2 * (1 + alpha**2) / beta)
        else:
            def opt_fun(x, a, b, sigma, weights):
                return weights * (1e6 * x[0] * (x[1] * a**2 - 2 * x[2] * a * b + (1 + x[2]**2) / x[1] * b**2) -
                                  (sigma * 1e3)**2)

            def opt_fun2(x, a, b, sigma, weights):
                return weights * (a**2 * x[0] - 2 * a * b * x[1] + b**2 * x[2] - sigma**2)

            if axis == "x":
                x0 = [self.eps_list[0], self.beta_list[0], self.alpha_list[0]]
            else:
                x0 = [self.eps_y_list[0], self.beta_y_list[0], self.alpha_y_list[0]]
            # # ldata = least_squares(opt_fun, x0, jac="2-point", args=(a, b, sigma),
            # #                       bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), gtol=1e-16, xtol=1e-16, ftol=1e-16)
            if charge is None:
                weights = 1.0
            else:
                s_q = np.sqrt(0.01 / np.log(2))
                weights = np.exp(-(charge / charge[0] - 1)**2 / s_q**2)
            # self.logger.debug("Weights: {0}".format(weights))
            ldata = least_squares(opt_fun, x0, jac="2-point", args=(a, b, sigma, weights),
                                  bounds=([0.2e-6 / self.gamma_energy, 0.0, -np.inf],
                                          [20e-6 / self.gamma_energy, 100.0, np.inf]))
            if ldata.success:
                eps = ldata.x[0]
                beta = ldata.x[1]
                alpha = ldata.x[2]
            else:
                self.logger.info("Direct twiss least squares failed. Attempting indirect")
                if axis == "x":
                    alpha0 = self.alpha_list[-1]
                    beta0 = self.beta_list[-1]
                    eps0 = self.eps_list[-1]
                    x0 = [eps0 * beta0, eps0 * alpha0, eps0 * (1 + alpha0**2) / beta0]
                else:
                    alpha0 = self.alpha_y_list[-1]
                    beta0 = self.beta_y_list[-1]
                    eps0 = self.eps_y_list[-1]
                    x0 = [eps0 * beta0, eps0 * alpha0, eps0 * (1 + alpha0**2) / beta0]
                ldata = least_squares(opt_fun2, x0, jac="2-point", args=(a, b, sigma, weights),
                                      bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
                eps2 = ldata.x[2] * ldata.x[0] - ldata.x[1]**2
                if eps2 < 0:
                    eps2 = eps0**2
                eps = np.sqrt(eps2)
                alpha = ldata.x[1] / eps
                beta = ldata.x[0] / eps

            self.logger.debug("Found twiss parameters:"
                              "\n alpha_{3}={0:.3f}\n beta_{3}={1:.3f}\n eps_n_{3}={2:.3f}".format(alpha, beta,
                                                                                       eps * self.gamma_energy * 1e6, axis))

        return alpha, beta, eps

    def set_target_ab(self, step, theta, r_maj, r_min, axis="x"):
        self.logger.debug("{0}: Determine new target a,b for axis {1}, step {2}".format(self, axis, step))
        if axis == "x":
            x_list = self.x_list
            target_sigma = self.target_sigma
            a_list = self.a_list
            b_list = self.b_list
        else:
            x_list = self.y_list
            target_sigma = self.target_sigma_y
            a_list = self.a_y_list
            b_list = self.b_y_list
        if self.algo == "const_size":
            if step < 3:

                # try:
                #     psi = self.psi_target[-1] - 0.01
                # except IndexError:
                #     psi = np.arccos((self.a_list[0] + self.b_list[0] * np.tan(theta)) * np.cos(theta) / r_maj)
                # target_a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
                # target_b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)

                c = x_list[-1] / target_sigma
                d_psi = 0.01
                # psi = np.arccos((self.a_list[-1] + self.b_list[-1] * np.tan(theta)) * np.cos(theta) / (c * r_maj))
                psi = self.get_psi(a_list[-1], b_list[-1], theta, c * r_maj, c * r_min)
                a1 = c * r_maj * np.cos(psi + d_psi) * np.cos(theta) - c * r_min * np.sin(psi + d_psi) * np.sin(theta)
                b1 = c * r_maj * np.cos(psi + d_psi) * np.sin(theta) + c * r_min * np.sin(psi + d_psi) * np.cos(theta)
                da = a1 - a_list[-1]
                db = b1 - b_list[-1]
                d_ab = 0.5 / np.sqrt(da**2 + db**2)
                self.logger.debug("da: {0:.3f}, db: {1:.3f}".format(da, db))
                target_a = a_list[-1] + d_ab * da
                target_b = b_list[-1] + d_ab * db
            else:
                a_min, a_max, b_min, b_max = self.get_ab_range(self.max_k)
                psi_v = np.linspace(0, 2 * np.pi, 5000)
                a, b = self.get_ab(psi_v, theta, r_maj, r_min)
                ind = np.all([a < a_max, a > a_min, b < b_max, b > b_min], axis=0)
                a_g = a[ind]
                b_g = b[ind]
                st = int(a_g.shape[0] / self.n_steps + 0.5)
                self.logger.debug("Found {0} values in range. Using every {1} value.".format(a_g.shape[0], st))
                self.logger.debug("Step {0} indexing a_g[::st] of length {1}.".format(step, a_g[::st].shape[0]))
                try:
                    target_a = a_g[::st][step]
                    target_b = b_g[::st][step]
                except IndexError as e:
                    self.logger.warning(e)
                    target_a = a_g[-1]
                    target_b = b_g[-1]
                psi = self.get_psi(target_a, target_b, theta, r_maj, r_min)

            self.psi_target.append(psi)
        else:
            target_a = 1
            target_b = 0
        # self.logger.debug("Target a, b = {0:.3f}, {1:.3f}".format(target_a, target_b))
        return target_a, target_b

    def solve_quads(self, target_a, target_b, target_a_y, target_b_y):
        t0 = time.time()
        self.logger.info("{0}: Solving new quad strengths for \n"
                         "Horizontal target a,b = {1:.3f}, {2:.3f}\n"
                         "Vertical target   a,b = {3:.3f}, {4:.3f}".format(self, target_a, target_b,
                                                                         target_a_y, target_b_y))

        def calc_sigma(alpha, beta, eps, a, b):
            sigma = np.sqrt(eps * (beta * a ** 2 - 2.0 * alpha * a * b + (1.0 + alpha ** 2) / beta * b ** 2))
            return sigma

        target_ap = np.array([target_a, target_b])
        th_ab = 0.2
        ind_p = ((self.A_lu[:, 0:2] - target_ap) ** 2).sum(1) < th_ab ** 2

        sigma_x = calc_sigma(self.alpha_list[-1], self.beta_list[-1], self.eps_list[-1],
                             self.A_lu[ind_p, 0], self.A_lu[ind_p, 1])
        sigma_y = calc_sigma(self.alpha_y_list[-1], self.beta_y_list[-1], self.eps_y_list[-1],
                             self.A_lu[ind_p, 2], self.A_lu[ind_p, 3])
        # Asi = np.stack((sigma_x.flatten()[ind_p] * sigma_y.flatten()[ind_p]), -1).reshape(-1, 1)
        try:
            Asi = np.stack((sigma_x.flatten() * sigma_y.flatten()), -1).reshape(-1, 1)
        except ValueError as e:
            self.logger.warning("Could not find quad values for a={0:.3f}, b={1:.3f}".format(target_a, target_b))
            return None
        target_asi = np.array([self.target_sigma * self.target_sigma_y]).reshape(-1, 1)
        ind_si = ((Asi - target_asi) ** 2).sum(-1)

        k_target = self.k_lu[ind_p, :][ind_si.argmin(), :]

        self.logger.debug("Found quad strengths: {0}".format(k_target))
        # self.logger.debug("{0}".format(res))
        self.logger.debug("Time: {0:.3f} s".format(time.time() - t0))
        return k_target

    def calc_response_matrix_v(self, quad_strengths, quad_positions, screen_position, axis="x"):
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = quad_positions[0]
        M = np.identity(2)
        if axis != "x":
            quad_strengths = -np.array(quad_strengths)
        for ind, quad in enumerate(quad_positions):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M = np.matmul(M_d, M)
            L = 0.2
            k = quad_strengths[..., ind]
            k_sqrt = np.sqrt(k * (1 + 0j))

            M_q = np.real(np.array([[np.cos(k_sqrt * L), L * sinc(L * k_sqrt)],
                                    [-k_sqrt * np.sin(k_sqrt * L), np.cos(k_sqrt * L)]]))
            M = np.matmul(np.moveaxis(M_q, (0, 1), (-2, -1)), M)
            s = quad + L
        drift = screen_position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M = np.matmul(M_d, M)
        return M

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
        if alpha != 0:
            theta = np.arctan((l1 - gamma) / alpha)
        else:
            theta = np.pi/2
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

    def get_ab_range(self, max_k, axis="x"):
        """
        Calculate range of a and b that is reachable given a maximum magnet k-value
        :param max_k: Magnet maximum k-value k = (-max_k, max_k)
        :return: Minimum a, maxmium a, minimum b, maximum b
        """
        def ab_fun(x, a=True, maxmin=False, axis="x"):
            if a:
                if maxmin:
                    return -self.calc_response_matrix(x, self.quad_list, self.screen.position, axis)[0, 0]
                else:
                    return self.calc_response_matrix(x, self.quad_list, self.screen.position, axis)[0, 0]
            else:
                if maxmin:
                    return -self.calc_response_matrix(x, self.quad_list, self.screen.position, axis)[0, 1]
                else:
                    return self.calc_response_matrix(x, self.quad_list, self.screen.position, axis)[0, 1]
        c = 0.8
        mag_r = Bounds(-c * max_k, c * max_k)
        x0 = np.array(self.quad_strength_list)
        self.logger.info("x0: {0}".format(x0))
        res = minimize(ab_fun, x0=x0, args=(True, False, axis), bounds=mag_r)
        a_min = self.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=x0, args=(True, True, axis), bounds=mag_r)
        a_max = self.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 0]
        res = minimize(ab_fun, x0=x0, args=(False, False, axis), bounds=mag_r)
        b_min = self.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 1]
        res = minimize(ab_fun, x0=x0, args=(False, True, axis), bounds=mag_r)
        b_max = self.calc_response_matrix(res.x, self.quad_list, self.screen.position)[0, 1]
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
            self.gamma_energy = 233e6 / 0.511e6
            self.quad_list = list()
            self.quad_list.append(SectionQuad("QB-01", 13.55, 0.2, "MAG-01", "CRQ-01", True))
            self.quad_list.append(SectionQuad("QB-02", 14.45, 0.2, "MAG-02", "CRQ-02", True))
            self.quad_list.append(SectionQuad("QB-03", 17.75, 0.2, "MAG-03", "CRQ-03", True))
            self.quad_list.append(SectionQuad("QB-04", 18.65, 0.2, "MAG-04", "CRQ-04", True))

            self.screen = SectionScreen("screen", 19.223, "liveviewer", "beamviewer", "limaccd", "screen")

            self.quad_strength_list = [-0.7, -0.3, -3.6, 2.3]  # 0.4 mm size

        elif section == "MS2":
            self.gamma_energy = 233e6 / 0.511e6

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


class MultiQuadTango(object):
    """
    Class that connects to tango device servers for magnets and screen camera and uses them to
    do a multi quad scan. The scan logic is from the MultiQuadManual class.
    """
    def __init__(self, n_steps=32):
        self.name = "MultiQuadTango"
        self.logger = logging.getLogger("Tango.{0}".format(self.name.upper()))
        self.logger.setLevel(logging.DEBUG)

        self.mq = MultiQuadLookup()
        self.magnet_names = ["i-ms1/mag/qb-01", "i-ms1/mag/qb-02", "i-ms1/mag/qb-03", "i-ms1/mag/qb-04"]
        # self.screen_name = "i-ms1/dia/scrn-01"
        self.camera_name = "lima/liveviewer/i-ms1-dia-scrn-01"
        self.sigma_target_x = 400e-6
        self.section = "MS1"
        self.n_steps = n_steps
        alpha0 = 0
        beta0 = 10
        eps_n_0 = 1e-6

        self.alpha = 15.0
        self.beta = 50.0
        self.eps_n = 5e-6
        self.alpha_y = 0.0
        self.beta_y = 30.0
        self.eps_n_y = 2e-6
        self.beamenergy = 233.0e6

        k0 = [2.0, 1.3, -3.8, 0.3]
        k0 = [-0.1, 0.1, -1.5, -1.4]

        self.image_list = list()
        self.image_p_list = list()

        self.magnet_devices = list()
        for mag in self.magnet_names:
            dev = pt.DeviceProxy("127.0.0.1:10000/{0}#dbase=no".format(mag))
            self.magnet_devices.append(dev)
            pos = dev.position
            self.logger.info("Connected to device {0}".format(mag))
        for ind, dev in enumerate(self.magnet_devices):
            dev.mainfieldcomponent = k0[ind]

        self.camera_device = pt.DeviceProxy("127.0.0.1:10002/{0}#dbase=no".format(self.camera_name))
        self.logger.info("Connected to device {0}".format(self.camera_name))
        camera_pos = self.camera_device.position
        self.px_cal = self.camera_device.pixel_cal
        self.roi = self.camera_device.roi
        self.roi = np.array([520, 400, 240, 240])
        self.camera_device.alpha = self.alpha
        self.camera_device.beta = self.beta
        self.camera_device.eps_n = self.eps_n * 1e6
        self.camera_device.alpha_y = self.alpha_y
        self.camera_device.beta_y = self.beta_y
        self.camera_device.eps_n_y = self.eps_n_y * 1e6
        self.camera_device.beamenergy = self.beamenergy * 1e-6
        sigma_x, sigma_y, image_p = self.process_image(self.camera_device.image, 0.95)
        self.sigma_target_x = sigma_x
        self.sigma_target_y = sigma_y
        self.charge = image_p.sum()

        self.base_filename = "multiquad"
        self.base_path = "..\\..\\data"
        self.pathname = None

        self.mq.start_scan(self.sigma_target_x, self.sigma_target_y, self.charge, self.section, self.n_steps,
                           alpha0, beta0, eps_n_0)
        self.current_step = 0

    def do_step(self, save=True):
        t0 = time.time()
        k_current = [dev.mainfieldcomponent for dev in self.magnet_devices]
        time.sleep(0.1)
        image = self.camera_device.image
        sigma_x, sigma_y, image_p = self.process_image(image, 0.95)
        self.image_list.append(image)
        self.image_p_list.append(image_p)
        charge = image_p.sum()
        self.logger.info("Step {0}: sigma_x={1:.3f} mm, sigma_y={2:.3f} mm".format(self.current_step,
                                                                                   sigma_x*1e3, sigma_y*1e3))
        if save:
            self.save_image(image, self.current_step, k_current)

        k_next = self.mq.scan_step(sigma_x, sigma_y, charge, k_current)
        if k_next is not None:
            for ind, dev in enumerate(self.magnet_devices):
                dev.mainfieldcomponent = k_next[ind]
        self.current_step += 1
        self.logger.debug("Step time: {0:.3} s".format(time.time() - t0))

    def process_image(self, image, keep_charge_ratio=0.95):
        t0 = time.time()
        image_roi = image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]

        # Median filter and background subtraction:
        image_p = medfilt2d(image_roi, 5)
        bkg = image_p[0:50, 0:50].max() * 2
        image_p[image_p < bkg] = 0

        # Find charge to keep:
        n_bins = np.unique(image_p.flatten()).shape[0]
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(image_p, n_bins)
        # Cumlative charge as function of pixel value:
        hq = (h[0]*h[1][:-1]).cumsum()
        # Normalize:
        hq = hq / np.float(hq.max())
        # Find index above which the desired charge ratio is:
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        # Background level:
        d = (h[1][1] - h[1][0])/2.0
        th_q = h[1][th_ind] - d
        image_p[image_p < th_q] = 0.0

        x_v = np.arange(image_p.shape[1])
        y_v = np.arange(image_p.shape[0])
        w0 = image_p.sum()
        x0 = (image_p.sum(0) * x_v).sum() / w0
        y0 = (image_p.sum(1) * y_v).sum() / w0
        sigma_x = self.px_cal * np.sqrt((image_p.sum(0) * (x_v - x0)**2).sum() / w0)
        sigma_y = self.px_cal * np.sqrt((image_p.sum(1) * (y_v - y0) ** 2).sum() / w0)
        self.logger.debug("Process image roi {0}x{1} time: {2:.3f} ms".format(image_roi.shape[1], image_roi.shape[0],
                                                                              1e3 * (time.time() - t0)))
        return sigma_x, sigma_y, image_p

    def do_scan(self, save=True):
        if save:
            s = "Multiquad_{0}_{1}".format(time.strftime("%Y-%m-%d_%H-%M-%S"), self.section)
            self.pathname = os.path.join(self.base_path, s)
            os.makedirs(self.pathname)
        for step in range(self.n_steps):
            self.do_step(save)

    def save_image(self, image, step, k_values):
        self.logger.info("Saving image {0}".format(step))
        s = "_".join(["{0:.3f}".format(k) for k in k_values])
        filename = "{0}_{1}_.png".format(step, s)
        full_name = os.path.join(self.pathname, filename)
        with open(full_name, "wb") as fh:
            try:
                write_png(fh, image.astype(np.uint16), filter_type=1)
            except Exception as e:
                self.logger.error("Image error: {0}".format(e))


if __name__ == "__main__":

    mt = MultiQuadTango()
    theta, r_maj, r_min = mt.mq.calc_ellipse(mt.alpha, mt.beta, mt.eps_n / (mt.beamenergy / 0.511e6), mt.sigma_target_x)
    theta_y, r_maj_y, r_min_y = mt.mq.calc_ellipse(mt.alpha_y, mt.beta_y, mt.eps_n_y / (mt.beamenergy / 0.511e6), mt.sigma_target_y)
    psi_v = np.linspace(0, 2 * np.pi, 1000)
    a_x, b_x = mt.mq.get_ab(psi_v, theta, r_maj, r_min)
    a_y, b_y = mt.mq.get_ab(psi_v, theta_y, r_maj_y, r_min_y)

