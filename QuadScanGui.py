# -*- coding:utf-8 -*-
"""
Created on May 18, 2018

@author: Filip Lindau
"""

from PyQt4 import QtGui, QtCore

import pyqtgraph as pq
import sys

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.DEBUG)

pq.graphicsItems.GradientEditorItem.Gradients['greyclip2'] = {
    'ticks': [(0.0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}
pq.graphicsItems.GradientEditorItem.Gradients['thermalclip'] = {
    'ticks': [(0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
              (1, (255, 255, 255, 255))], 'mode': 'rgb'}


class QuadScanGui(QtGui.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """

    def __init__(self, parent=None):
        root.debug("Init")
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'QuadScan')

        self.raw_image_widget = None
        self.proc_image_widget = None
        self.scan_image_widget = None
        self.scan_plot_widget = None
        self.roi_widget = None

        self.quad_select_combobox = None
        self.quad_start_spinbox = None
        self.quad_stop_spinbox = None
        self.scan_steps_spinbox = None
        self.screen_select_combobox = None
        self.roi_x_spinbox = None
        self.roi_y_spinbox = None
        self.roi_w_spinbox = None
        self.roi_h_spinbox = None

        self.image_threshold_spinbox = None

        self.eps_label = None
        self.beta_label = None
        self.gamma_label = None
        self.alpha_label = None

    def setup_layout(self):
        """
        Setup GUI layout and set stored settings
        :return:
        """
        # Plotting widgets:
        plt1 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.raw_image_widget = pq.ImageView(view=plt1)
        self.raw_image_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.raw_image_widget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.raw_image_widget.getView().setAspectLocked(False)
        h = self.raw_image_widget.getHistogramWidget()
        h.item.sigLevelChangeFinished.connect(self.update_image_threshold)
        self.roi_widget = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.roi_widget.sigRegionChangeFinished.connect(self.update_roi)
        self.roi_widget.blockSignals(True)

        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        main_layout = QtGui.QVBoxLayout(self)

        input_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(input_layout)
        device_layout = QtGui.QGridLayout()
        input_layout.addLayout(device_layout)

        graphics_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(graphics_layout)
        graphics_layout.addWidget(self.raw_image_widget)

    def update_image_threshold(self):
        """
        Set the threshold used when processing image for beam emittance calculations
        :return:
        """
        pass

    def update_roi(self):
        """
        Update the roi in the raw image used for beam emittance calculations
        :return:
        """
        pass


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    myapp.show()
    sys.exit(app.exec_())
