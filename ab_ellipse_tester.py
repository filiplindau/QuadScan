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

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setup_layout()
        self.update_ellipse()

    def setup_layout(self):
        # self.ui.widget = pq.PlotWidget()
        self.ui.alpha_min_spinbox.editingFinished.connect(self.update_limits)
        self.ui.alpha_slider.valueChanged.connect(self.update_ellipse)
        self.ui.beta_slider.valueChanged.connect(self.update_ellipse)
        self.ui.eps_slider.valueChanged.connect(self.update_ellipse)
        self.ui.sigma_slider.valueChanged.connect(self.update_ellipse)
        self.ui.energy_slider.valueChanged.connect(self.update_ellipse)

        self.ellipse_plot = self.ui.widget.plot()
        self.ellipse_plot.setPen((200, 150, 50), width=2.0)
        self.ui.widget.showGrid(True, True)

    def update_limits(self):
        pass

    def update_ellipse(self):
        alpha = self.ui.alpha_min_spinbox.value() + 0.01 * self.ui.alpha_slider.value() * (self.ui.alpha_max_spinbox.value() - self.ui.alpha_min_spinbox.value())
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
        theta, r_maj, r_min = self.calc_ellipse(alpha, beta, eps, sigma, energy)
        psi = np.linspace(0, 2*np.pi, 1000)
        self.a_v = r_maj * np.cos(psi) * np.cos(theta) - r_min * np.sin(psi) * np.sin(theta)
        self.b_v = r_maj * np.cos(psi) * np.sin(theta) + r_min * np.sin(psi) * np.cos(theta)
        self.ellipse_plot.setData(x=self.a_v, y=self.b_v)
        self.ui.widget.update()

    def calc_ellipse(self, alpha, beta, eps, sigma, energy):
        logger.debug("alpha {0}, beta {1}, eps{2}, sigma {3}, energy {4}".format(alpha, beta, eps, sigma, energy))
        my = sigma**2.0 / (eps * energy / 0.511)
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = ABEllipseTester()
    logger.info("QuadScanGui object created")
    myapp.show()
    logger.info("App show")
    sys.exit(app.exec_())
    logger.info("App exit")
