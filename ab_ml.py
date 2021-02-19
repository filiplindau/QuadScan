"""
Create a neural network model of quad settings vs a-b values (x and y).
"""
import numpy as np
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import ghalton
import time

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


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


def _generate_halton(n_points, n_dim=4, seed=-1):
    """
    Genrate Halton sequence using the ghalton package.
    If the seed value is not None, the sequencer is seeded and reset.
    :param n_points: Number of points to generate
    :param n_dim: Number of dimensions in the generated sequence
    :param seed: Seed value for the generator. See ghalton documentation
    for details. Use seed=-1 for optimized DeRainville2012 sequence permutations.
    :return: Numpy array of shape [n_points, n_dim]
    """
    halton_sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:n_dim])
    if seed is not None:
        if seed == -1:
            seed = ghalton.EA_PERMS
        halton_sequencer.seed(seed)
    p = np.array(halton_sequencer.get(int(n_points)))
    return p


def generate_dataset(n_samples, k_max=8.0, k_min=-8.0, section="MS1"):
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

    n_q = quad_positions.shape[0]
    p = _generate_halton(n_samples, n_q, -1)
    quad_strengths = k_min + (k_max - k_min) * p
    t0 = time.time()
    M = calc_response_matrix(quad_strengths, quad_positions, screen_position, "x")
    ax = M[..., 0, 0]
    bx = M[..., 0, 1]
    logger.info("Time x: {0}".format(time.time()-t0))
    t1 = time.time()
    M = calc_response_matrix(quad_strengths, quad_positions, screen_position, "y")
    ay = M[..., 0, 0]
    by = M[..., 0, 1]
    logger.info("Time y: {0}".format(time.time()-t1))
    A = np.stack((ax, bx, ay, by), -1).reshape(-1, n_q)
    k = quad_strengths.reshape(-1, n_q)
    return A, k, p


def generate_ds2(section="MS1"):
    import pickle
    with open("{0}_k.pkl".format(section), "rb") as f:
        data = pickle.load(f)
        km = data[0]
        ab = data[1]
        av = data[2]
        bv = data[3]
    ab_data = list()
    k_data = list()
    for ia, a in enumerate(av):
        for ib, b in enumerate(bv):
            m = ab[ia][ib]
            mab = np.tile([av[ia], bv[ib]], [m.shape[0], 1])
            try:
                ab_data.append(np.concatenate((m, mab), 1))
                k_data.append(km[ia][ib])
            except np.AxisError:
                pass
    return np.concatenate(ab_data, 0), np.concatenate(k_data, 0)


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

# try:
#     A
# except NameError:
#     A, k, p = generate_dataset(1000000, 8.0, -8.0, "MS1")
# X = A[:, :2]
# y = p
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
# scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_test = scaling.transform(X_test)

# mlp = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32, 32, 8), activation="tanh", solver="sgd", max_iter=200,
#                    verbose=True, early_stopping=True, tol=1e-3)
# mlp.fit(X_train, y_train)

# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)

# svr = svm.NuSVR(nu=0.5, C=1.0, kernel="rbf", verbose=True)
# svr.fit(X_train, y_train)

# predict_train = svr.predict(X_train)
# predict_test = svr.predict(X_test)

ab_data, k_data = generate_ds2("MS1")
X = ab_data[:1000000, :]
y = k_data[:1000000, :]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
y_train = scaling.transform(y_train)
y_test = scaling.transform(y_test)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

model = nn.Sequential(nn.Linear(X_train.shape[1], 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Linear(256, 32),
                      nn.ReLU(),
                      nn.Linear(32, y_train.shape[1]),
                      nn.Tanh()).to("cuda:0")
logger.debug("Model created")

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = nn.MSELoss()
logger.debug("Optimizer")

inputs = Variable(X_train).to("cuda:0")
outputs = Variable(y_train).to("cuda:0")
logger.debug("Input and output created")

for epoch in range(10000):
    y_pred = model(inputs)
    loss = loss_func(y_pred, outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 50 == 0:
        logger.info("Epoch {0}: loss {1}".format(epoch, loss.item()))

logger.info("\nTrain set MSE: {0}\n"
            "Test set MSE:  {1}\n".format(loss_func(model(inputs), outputs).item(),
                                          loss_func(model(X_test.to("cuda:0")), y_test.to("cuda:0")).item()))

kt = model(torch.from_numpy(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, -1)).float().to("cuda:0")).\
         detach().cpu().numpy()*16-8
calc_response_matrix(kt, quad_positions, screen_position)
