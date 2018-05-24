# -*- coding:utf-8 -*-
"""
Created on May 18, 2018

@author: Filip Lindau
"""

from PyQt4 import QtGui, QtCore

import pyqtgraph as pq
import sys
import numpy as np
from QuadScanController import QuadScanController
from QuadScanState import StateDispatcher
from quadscan_ui import Ui_QuadScanDialog

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

        self.current_state = "unknown"
        self.last_load_dir = "."

        self.ui = Ui_QuadScanDialog()
        self.ui.setupUi(self)

        self.setup_layout()
        self.controller = QuadScanController()
        self.controller.add_state_notifier(self.change_state)

        self.state_dispatcher = StateDispatcher(self.controller)
        self.state_dispatcher.start()

    def setup_layout(self):
        """
        Setup GUI layout and set stored settings
        :return:
        """
        # Plotting widgets:
        plt1 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.ui.image_raw_widget = pq.ImageView(view=plt1)
        # self.ui.image_raw_widget.histogram.gradient.loadPreset('thermalclip')
        self.ui.image_raw_widget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.ui.image_raw_widget.getView().setAspectLocked(False)
        self.ui.image_raw_widget.setData(np.random.random((64, 64)))
        h = self.ui.image_raw_widget.getHistogramWidget()
        h.item.sigLevelChangeFinished.connect(self.update_image_threshold)
        self.ui.roi_widget = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.ui.roi_widget.sigRegionChangeFinished.connect(self.update_roi)
        self.ui.roi_widget.blockSignals(True)

        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))

        # Signal connections
        self.ui.k_slider.valueChanged.connect(self.update_image_selection)
        self.ui.image_slider.valueChanged.connect(self.update_image_selection)

        # Data storage
        self.ui.load_data_button.clicked.connect(self.load_data)

        self.last_load_dir = self.settings.value("load_path", ".", type=str)

    def change_state(self, new_state, new_status=None):
        root.info("Change state: {0}, status {1}".format(new_state, new_status))
        self.ui.state_label.setText(QtCore.QString(new_state))
        if new_status is not None:
            self.ui.status_label.setText(QtCore.QString(new_status))
        if self.current_state == "load" and new_state != "load":
            self.update_parameter_data()
        self.current_state = new_state

    def update_parameter_data(self):
        root.info("Updating parameters")
        quad_length = self.controller.get_parameter("scan", "quad_length")
        root.debug("quad_length: {0}".format(quad_length))
        self.ui.quad_length_label.setText(str(quad_length))
        self.ui.quad_screen_distance_label.setText(str(self.controller.get_parameter("scan", "quad_screen_distance")))
        self.ui.energy_spinbox.setValue(self.controller.get_parameter("scan", "beam_energy"))
        self.ui.k_start_spinbox.setValue(self.controller.get_parameter("scan", "k_min"))
        self.ui.k_end_spinbox.setValue(self.controller.get_parameter("scan", "k_max"))
        self.ui.k_number_values_spinbox.setValue(self.controller.get_parameter("scan", "num_k_values"))
        self.ui.images_number_spinbox.setValue(self.controller.get_parameter("scan", "num_shots"))
        self.ui.quad_select_edit.setText(self.controller.get_parameter("scan", "quad_name"))
        self.ui.screen_select_edit.setText(self.controller.get_parameter("scan", "screen_name"))

        self.ui.k_slider.setMaximum(self.controller.get_parameter("scan", "num_k_values") - 1)
        self.ui.image_slider.setMaximum(self.controller.get_parameter("scan", "num_shots") - 1)
        self.update_image_selection()

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

    def update_image_selection(self):
        """
        Update the image selected by the sliders.
        :return:
        """
        image_ind = self.ui.image_slider.value()
        k_ind = self.ui.k_slider.value()
        raw_data = self.controller.get_result("scan", "raw_data")
        root.debug("Updating image to {0}:{1} in an array of {2}:{3}".format(k_ind, image_ind,
                                                                             len(raw_data), len(raw_data[0])))
        pic = raw_data[k_ind][image_ind]
        self.ui.image_raw_widget.setImage(pic)

    def load_data(self):
        root.info("Loading data")
        load_dir = QtGui.QFileDialog.getExistingDirectory(self, "Select directory", self.last_load_dir)
        self.last_load_dir = load_dir
        root.debug("Loading from directory {0}".format(load_dir))
        self.controller.set_parameter("load", "path", str(load_dir))
        self.state_dispatcher.send_command("load")

    def closeEvent(self, event):
        """
        Closing the applications. Stopping threads and saving the settings.
        :param event:
        :return:
        """
        self.state_dispatcher.stop()
        self.settings.setValue("load_path", self.last_load_dir)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    myapp.show()
    sys.exit(app.exec_())
