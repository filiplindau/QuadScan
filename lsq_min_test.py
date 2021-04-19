from scipy.optimize import minimize
import numpy as np

def fun(x, M, s):
    y = np.dot(M, x)
    return np.sum((y - s**2)**2)


alpha0 = -4.0
beta0 = 10.0
eps0 = 1e-6 / 469.7

x0 = beta0 *  eps0
x1 = alpha0 * eps0
x2 = (eps0**2 + x1**2) / x0
xi = np.array([x0, x1, x2])

cons = ({"type": "ineq", "fun": lambda x: x[0]*x[2]-x[1]**2})
bnds = ((0, None), (None, None), (0, None))

res = minimize(fun, xi, args=(M, s), method="SLSQP", bounds=bnds, constraints=cons, tol=1e-8)
