"""
GUI to visualize and test the size of the a-b ellipse for different twiss and quad parameters

Created 2020-02-05

@author: Filip Lindau

"""

import numpy as np
import pyqtgraph as pq
import sys
from PyQt5 import QtWidgets
from ab_ellipse_tester_ui import Ui_Dialog
import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


class ABEllipseTester(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.a_v = None
        self.b_v = None
        self.ellipse_plot = None
        self.quad_plot_x = None
        self.quad_plot_y = None

        self.quad_pos = np.array([13.55, 14.45, 17.75, 18.65])

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setup_layout()
        self.update_ellipse()
        self.update_quads()

    def setup_layout(self):
        self.ui.alpha_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.alpha_max_spinbox.editingFinished.connect(self.update_limits)
        self.ui.beta_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.beta_max_spinbox.editingFinished.connect(self.update_limits)
        self.ui.eps_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.eps_max_spinbox.editingFinished.connect(self.update_limits)
        self.ui.sigma_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.sigma_max_spinbox.editingFinished.connect(self.update_limits)
        self.ui.energy_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.energy_max_spinbox.editingFinished.connect(self.update_limits)
        self.ui.alpha_slider.valueChanged.connect(self.update_ellipse)
        self.ui.beta_slider.valueChanged.connect(self.update_ellipse)
        self.ui.eps_slider.valueChanged.connect(self.update_ellipse)
        self.ui.sigma_slider.valueChanged.connect(self.update_ellipse)
        self.ui.energy_slider.valueChanged.connect(self.update_ellipse)
        self.ui.alpha_spinbox.editingFinished.connect(self.update_limits)
        self.ui.beta_spinbox.editingFinished.connect(self.update_limits)
        self.ui.eps_spinbox.editingFinished.connect(self.update_limits)
        self.ui.sigma_spinbox.editingFinished.connect(self.update_limits)
        self.ui.energy_spinbox.editingFinished.connect(self.update_limits)

        self.ui.k1_min_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k1_max_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k2_min_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k2_max_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k3_min_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k3_max_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k4_min_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k4_max_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.screen_min_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.screen_max_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k1_slider.valueChanged.connect(self.update_quads)
        self.ui.k2_slider.valueChanged.connect(self.update_quads)
        self.ui.k3_slider.valueChanged.connect(self.update_quads)
        self.ui.k4_slider.valueChanged.connect(self.update_quads)
        self.ui.screen_slider.valueChanged.connect(self.update_quads)
        self.ui.k1_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k2_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k3_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.k4_spinbox.editingFinished.connect(self.update_limits_quads)
        self.ui.screen_spinbox.editingFinished.connect(self.update_limits_quads)

        self.ui.widget.showGrid(True, True)
        self.ui.widget.setAspectLocked(True)
        self.ui.widget.setLabel("bottom", "a")
        self.ui.widget.setLabel("left", "b")
        self.ui.widget.addLegend()
        self.ellipse_plot = self.ui.widget.plot()
        self.ellipse_plot.setPen((200, 150, 50), width=2.0)
        self.quad_plot_x = self.ui.widget.plot(pen=None, symbol="o", size=10,
                                               symbolBrush=pq.mkBrush(30, 180, 50), symbolPen=None, name="x")
        self.quad_plot_y = self.ui.widget.plot(pen=None, symbol="d", size=10,
                                               symbolBrush=pq.mkBrush(30, 100, 230), symbolPen=None, name="y")

    def update_limits(self):
        self.ui.alpha_slider.blockSignals(True)
        self.ui.beta_slider.blockSignals(True)
        self.ui.eps_slider.blockSignals(True)
        self.ui.sigma_slider.blockSignals(True)
        self.ui.energy_slider.blockSignals(True)

        x = self.ui.alpha_spinbox.value()
        x_min = self.ui.alpha_min_spinbox.value()
        x_max = self.ui.alpha_max_spinbox.value()
        self.ui.alpha_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))

        x = self.ui.beta_spinbox.value()
        x_min = self.ui.beta_min_spinbox.value()
        x_max = self.ui.beta_max_spinbox.value()
        self.ui.beta_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))

        x = self.ui.eps_spinbox.value()
        x_min = self.ui.eps_min_spinbox.value()
        x_max = self.ui.eps_max_spinbox.value()
        self.ui.eps_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))

        x = self.ui.sigma_spinbox.value()
        x_min = self.ui.sigma_min_spinbox.value()
        x_max = self.ui.sigma_max_spinbox.value()
        self.ui.sigma_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))

        x = self.ui.energy_spinbox.value()
        x_min = self.ui.energy_min_spinbox.value()
        x_max = self.ui.energy_max_spinbox.value()
        self.ui.energy_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))

        self.ui.alpha_slider.blockSignals(False)
        self.ui.beta_slider.blockSignals(False)
        self.ui.eps_slider.blockSignals(False)
        self.ui.sigma_slider.blockSignals(False)
        self.ui.energy_slider.blockSignals(False)
        self.update_ellipse()

    def update_limits_quads(self):
        x = self.ui.k1_spinbox.value()
        x_min = self.ui.k1_min_spinbox.value()
        x_max = self.ui.k1_max_spinbox.value()
        self.ui.k1_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))
        x = self.ui.k2_spinbox.value()
        x_min = self.ui.k2_min_spinbox.value()
        x_max = self.ui.k2_max_spinbox.value()
        self.ui.k2_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))
        x = self.ui.k3_spinbox.value()
        x_min = self.ui.k3_min_spinbox.value()
        x_max = self.ui.k3_max_spinbox.value()
        self.ui.k3_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))
        x = self.ui.k4_spinbox.value()
        x_min = self.ui.k4_min_spinbox.value()
        x_max = self.ui.k4_max_spinbox.value()
        self.ui.k4_slider.setValue(np.round(100 * (x - x_min) / (x_max - x_min)))
        x = self.ui.screen_spinbox.value()
        x_min = self.ui.screen_min_spinbox.value()
        x_max = self.ui.screen_max_spinbox.value()
        self.ui.screen_slider.setValue(int(100 * (x - x_min) / (x_max - x_min)))

    def update_ellipse(self):
        alpha = self.ui.alpha_min_spinbox.value() + 0.01 * self.ui.alpha_slider.value() * (
                self.ui.alpha_max_spinbox.value() - self.ui.alpha_min_spinbox.value())
        beta = self.ui.beta_min_spinbox.value() + 0.01 * self.ui.beta_slider.value() * (
                    self.ui.beta_max_spinbox.value() - self.ui.beta_min_spinbox.value())
        eps = self.ui.eps_min_spinbox.value() + 0.01 * self.ui.eps_slider.value() * (
                    self.ui.eps_max_spinbox.value() - self.ui.eps_min_spinbox.value())
        sigma = self.ui.sigma_min_spinbox.value() + 0.01 * self.ui.sigma_slider.value() * (
                    self.ui.sigma_max_spinbox.value() - self.ui.sigma_min_spinbox.value())
        energy = self.ui.energy_min_spinbox.value() + 0.01 * self.ui.energy_slider.value() * (
                    self.ui.energy_max_spinbox.value() - self.ui.energy_min_spinbox.value())
        self.ui.alpha_spinbox.setValue(alpha)
        self.ui.beta_spinbox.setValue(beta)
        self.ui.eps_spinbox.setValue(eps)
        self.ui.sigma_spinbox.setValue(sigma)
        self.ui.energy_spinbox.setValue(energy)
        theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps * 1e-6, sigma * 1e-3, energy)
        psi = np.linspace(0, 2*np.pi, 1000)
        self.a_v = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
        self.b_v = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
        self.ellipse_plot.setData(x=self.a_v, y=self.b_v)
        self.ui.widget.update()
        self.update_quads()

    def update_quads(self):
        self.ui.k1_spinbox.setValue(self.ui.k1_min_spinbox.value() + 0.01 * self.ui.k1_slider.value() * (
                self.ui.k1_max_spinbox.value() - self.ui.k1_min_spinbox.value()))
        self.ui.k2_spinbox.setValue(self.ui.k2_min_spinbox.value() + 0.01 * self.ui.k2_slider.value() * (
                self.ui.k2_max_spinbox.value() - self.ui.k2_min_spinbox.value()))
        self.ui.k3_spinbox.setValue(self.ui.k3_min_spinbox.value() + 0.01 * self.ui.k3_slider.value() * (
                self.ui.k3_max_spinbox.value() - self.ui.k3_min_spinbox.value()))
        self.ui.k4_spinbox.setValue(self.ui.k4_min_spinbox.value() + 0.01 * self.ui.k4_slider.value() * (
                self.ui.k4_max_spinbox.value() - self.ui.k4_min_spinbox.value()))
        self.ui.screen_spinbox.setValue(self.ui.screen_min_spinbox.value() + 0.01 * self.ui.screen_slider.value() * (
                self.ui.screen_max_spinbox.value() - self.ui.screen_min_spinbox.value()))

        M_x, M_y = self.calc_response_matrix()
        a_x = M_x[0, 0]
        b_x = M_x[0, 1]
        self.quad_plot_x.setData(x=[a_x], y=[b_x])
        a_y = M_y[0, 0]
        b_y = M_y[0, 1]
        self.quad_plot_y.setData(x=[a_y], y=[b_y])
        alpha = self.ui.alpha_spinbox.value()
        beta = self.ui.beta_spinbox.value()
        eps = self.ui.eps_spinbox.value() * 1e-6
        energy = self.ui.energy_spinbox.value() / 0.511
        sigma_x = np.sqrt(
            eps / energy * (beta * a_x ** 2 - 2.0 * alpha * a_x * b_x + (1.0 + alpha ** 2) / beta * b_x ** 2))
        sigma_y = np.sqrt(
            eps / energy * (beta * a_y ** 2 - 2.0 * alpha * a_y * b_y + (1.0 + alpha ** 2) / beta * b_y ** 2))
        self.ui.sigma_x_label.setText("{0:.2f} mm".format(sigma_x * 1e3))
        self.ui.sigma_y_label.setText("{0:.2f} mm".format(sigma_y * 1e3))

    def calc_response_matrix(self):
        # self.logger.debug("{0}: Calculating new response matrix".format(self))
        s = self.quad_pos[0]
        quad_strengths = [self.ui.k1_spinbox.value(), self.ui.k2_spinbox.value(),
                          self.ui.k3_spinbox.value(), self.ui.k4_spinbox.value()]
        screen_position = self.ui.screen_spinbox.value() + 0.2 + self.quad_pos[3]
        M_x = np.identity(2)
        M_y = np.identity(2)
        for ind, quad in enumerate(self.quad_pos):
            # self.logger.debug("Position s: {0} m".format(s))
            drift = quad - s
            M_d = np.array([[1.0, drift], [0.0, 1.0]])
            M_x = np.matmul(M_d, M_x)
            M_y = np.matmul(M_d, M_y)
            L = 0.2
            k = quad_strengths[ind]
            if k != 0:
                k_sqrt_x = np.sqrt(k * (1 + 0j))
                k_sqrt_y = np.sqrt(-k * (1 + 0j))

                M_q_x = np.real(np.array([[np.cos(k_sqrt_x * L), np.sin(k_sqrt_x * L) / k_sqrt_x],
                                          [-k_sqrt_x * np.sin(k_sqrt_x * L), np.cos(k_sqrt_x * L)]]))
                M_q_y = np.real(np.array([[np.cos(k_sqrt_y * L), np.sin(k_sqrt_y * L) / k_sqrt_y],
                                          [-k_sqrt_y * np.sin(k_sqrt_y * L), np.cos(k_sqrt_y * L)]]))
            else:
                M_q_x = np.array([[1, L], [0, 1]])
                M_q_y = np.array([[1, L], [0, 1]])
            M_x = np.matmul(M_q_x, M_x)
            M_y = np.matmul(M_q_y, M_y)
            s = quad + L
        drift = screen_position - s
        M_d = np.array([[1.0, drift], [0.0, 1.0]])
        M_x = np.matmul(M_d, M_x)
        M_y = np.matmul(M_d, M_y)
        return M_x, M_y

    def calc_ellipse(self, alpha, beta, eps, sigma, energy):
        logger.debug("alpha {0}, beta {1}, eps{2}, sigma {3}, energy {4}".format(alpha, beta, eps, sigma, energy))
        my = sigma**2.0 / (eps / (energy / 0.511))
        gamma = (1.0 + alpha**2) / beta
        m11 = beta
        m12 = -alpha
        m22 = gamma
        l1 = ((m11 + m22) + np.sqrt((m11 - m22) ** 2 + 4.0 * m12 ** 2)) / 2
        l2 = ((m11 + m22) - np.sqrt((m11 - m22) ** 2 + 4.0 * m12 ** 2)) / 2
        r_minor = np.sqrt(my / l1)
        r_major = np.sqrt(my / l2)
        try:
            theta = np.arctan((l1 - gamma) / alpha)
        except ZeroDivisionError:
            theta = np.pi / 2
        return theta, r_major, r_minor


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = ABEllipseTester()
    logger.info("QuadScanGui object created")
    myapp.show()
    logger.info("App show")
    sys.exit(app.exec_())
    logger.info("App exit")
