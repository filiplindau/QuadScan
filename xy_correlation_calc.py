import numpy as np

def xy_correlation_calc(target_a, target_b):  # , A_lu, k_lu, alpha_list, beta_list, eps_list, target_sigma, target_sigma_y):
    def calc_sigma(alpha, beta, eps, a, b):
        sigma = np.sqrt(eps * (beta * a ** 2 - 2.0 * alpha * a * b + (1.0 + alpha ** 2) / beta * b ** 2))
        return sigma

    ay_list = list()
    by_list = list()
    for j in range(target_a.shape[0]):
        target_ap = np.array([target_a[j], target_b[j]])
        th_ab = 0.2
        ind_p = ((A_lu[:, 0:2] - target_ap) ** 2).sum(1) < th_ab ** 2

        sigma_x = calc_sigma(alpha_list[-1], beta_list[-1], eps_list[-1],
                             A_lu[ind_p, 0], A_lu[ind_p, 1])
        sigma_y = calc_sigma(alpha_y_list[-1], beta_y_list[-1], eps_y_list[-1],
                             A_lu[ind_p, 2], A_lu[ind_p, 3])
        Asi = np.stack((sigma_x.flatten() * sigma_y.flatten()), -1).reshape(-1, 1)
        target_asi = np.array([target_sigma * target_sigma_y]).reshape(-1, 1)
        ind_si = ((Asi - target_asi) ** 2).sum(-1)

        k_target = k_lu[ind_p, :]
        isis = ind_si.argsort()

        quad_positions = np.array([13.55, 14.45, 17.75, 18.65])
        screen_position = 19.223
        axis = "y"
        quad_strengths = k_target[isis[0:np.minimum(5, len(isis))]]

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
        ay_list.append(M[0, 0])
        by_list.append(M[0, 1])

    return ay_list, by_list