"""
Create a neural network model of quad settings vs a-b values (x and y).
"""
import logging
import numpy as np

def sinc(x):
    y = np.sin(x) / x
    y[np.isnan(y)] = 1.0
    return y


def calc_response_matrix(quad_strengths, quad_positions, screen_position, axis="x"):
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


def calc_sigma(alpha, beta, eps, a, b):
    sigma = np.sqrt(eps * (beta * a ** 2 - 2.0 * alpha * a * b + (1.0 + alpha ** 2) / beta * b ** 2))
    return sigma


def calc_ellipse(alpha, beta, eps, sigma):
    logger.debug("Twiss indata: alpha={0:.3f}, beta={1:.3f}, eps={2:.3g}, sigma={3:.3g}".format(alpha, beta, eps, sigma))
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
    logger.debug("Result: theta={0:.3f}, r_maj={1:.3f}, r_min={2:.3f}".format(theta, r_major, r_minor))
    return theta, r_major, r_minor


def get_ab(psi, theta, r_maj, r_min):
    a = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
    b = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
    return a, b


alpha_x = -5.0
beta_x = 17.0
eps_x = 1e-6 / (233.0/0.511)

alpha_y = 5.0
beta_y = 23.0
eps_y = 2e-6 / (233.0/0.511)

section = "MS1"
if section == "MS1":
    screen_position = 19.223
    quad_positions = np.array([13.55, 14.45, 17.75, 18.65])
elif section == "MS2":
    screen_position = 38.445
    quad_positions = np.array([33.52, 34.62, 35.62, 37.02])
elif section == "MS3":
    screen_position = 282.456
    quad_positions = np.array([275.719, 277.719, 278.919, 281.119, 281.619, 282.019])
else:
    screen_position = 19.223
    quad_positions = np.array([13.55, 14.45, 17.75, 18.65])

