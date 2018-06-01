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
import threading
import time

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

        self.line_x_plot = None
        self.line_y_plot = None
        self.cent_plot = None
        self.sigma_x_plot = None
        self.fit_x_plot = None

        self.gui_lock = threading.Lock()

        self.ui = Ui_QuadScanDialog()
        self.ui.setupUi(self)

        self.controller = QuadScanController()
        # self.controller.add_state_notifier(self.change_state)
        # self.controller.add_progress_notifier(self.change_progress)
        self.setup_layout()

        self.state_dispatcher = StateDispatcher(self.controller)
        self.state_dispatcher.start()

    def setup_layout(self):
        """
        Setup GUI layout and set stored settings
        :return:
        """
        # Plotting widgets:
        self.ui.image_raw_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.ui.image_raw_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ui.image_raw_widget.getView().setAspectLocked(False)
        self.ui.image_raw_widget.setImage(np.random.random((64, 64)))
        self.ui.image_raw_widget.ui.roiBtn.hide()
        self.ui.image_raw_widget.ui.menuBtn.hide()
        h = self.ui.image_raw_widget.getHistogramWidget()
        # h.item.sigLevelChangeFinished.connect(self.update_image_threshold)
        self.ui.image_raw_widget.roi.show()
        self.ui.image_raw_widget.roi.sigRegionChangeFinished.connect(self.update_roi)
        self.ui.image_raw_widget.roi.blockSignals(True)
        self.ui.image_raw_widget.roi.setPos((0, 0))
        self.ui.image_raw_widget.roi.setSize((64, 64))
        self.ui.image_raw_widget.roi.blockSignals(False)

        self.ui.image_proc_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.ui.image_proc_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ui.image_proc_widget.getView().setAspectLocked(False)
        self.ui.image_proc_widget.setImage(np.random.random((64, 64)))
        self.ui.image_proc_widget.ui.roiBtn.hide()
        self.ui.image_proc_widget.ui.menuBtn.hide()
        self.cent_plot = pq.PlotDataItem()
        self.ui.image_proc_widget.addItem(self.cent_plot)

        self.line_x_plot = self.ui.lineout_widget.plot()
        self.line_x_plot.setPen((200, 25, 10))
        self.line_y_plot = self.ui.lineout_widget.plot()
        self.line_y_plot.setPen((10, 200, 25))
        self.ui.lineout_widget.setLabel("bottom", "Line coord", "px")
        self.ui.lineout_widget.showGrid(True)

        self.sigma_x_plot = self.ui.fit_widget.plot()
        self.sigma_x_plot.setPen((10, 200, 25))
        self.fit_x_plot = self.ui.fit_widget.plot()
        self.fit_x_plot.setPen(pq.mkPen(color=(180, 180, 250), width=2))
        self.ui.fit_widget.setLabel("bottom", "K", "1/m^2")
        self.ui.fit_widget.setLabel("left", "sigma^2", "m^2")

        # This is to make sure . is the decimal character
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))

        # Signal connections
        self.ui.threshold_spinbox.editingFinished.connect(self.update_image_threshold)
        self.ui.k_slider.valueChanged.connect(self.update_image_selection)
        self.ui.image_slider.valueChanged.connect(self.update_image_selection)
        self.controller.progress_signal.connect(self.change_progress)
        self.controller.processing_done_signal.connect(self.update_image_selection)
        self.controller.fit_done_signal.connect(self.update_fit_data)
        self.controller.state_change_signal.connect(self.change_state)
        self.sigma_x_plot.sigPointsClicked.connect(self.points_clicked)

        # Data storage
        self.ui.load_data_button.clicked.connect(self.load_data)

        self.last_load_dir = self.settings.value("load_path", ".", type=str)

        # Geometry setup
        window_pos_x = self.settings.value('window_pos_x', 100, type=int)
        window_pos_y = self.settings.value('window_pos_y', 100, type=int)
        window_size_w = self.settings.value('window_size_w', 1100, type=int)
        window_size_h = self.settings.value('window_size_h', 800, type=int)
        if window_pos_x < 50:
            window_pos_x = 50
        if window_pos_y < 50:
            window_pos_y = 50
        self.setGeometry(window_pos_x, window_pos_y, window_size_w, window_size_h)

    def change_state(self, new_state, new_status=None):
        root.info("Change state: {0}, status {1}".format(new_state, new_status))
        self.ui.state_label.setText(QtCore.QString(new_state))
        if new_status is not None:
            self.ui.status_label.setText(QtCore.QString(new_status))
        if self.current_state == "load" and new_state != "load":
            self.update_parameter_data()
        self.current_state = new_state

    def change_progress(self, new_progress):
        root.info("Changing progress to {0}".format(new_progress))
        p = np.minimum(100, int(100*new_progress))
        p = np.maximum(0, p)
        root.debug("p: {0}".format(p))
        with self.gui_lock:
            self.ui.operation_progressbar.setValue(p)

    def update_parameter_data(self):
        root.info("Updating parameters")
        quad_length = self.controller.get_parameter("scan", "quad_length")
        root.debug("quad_length: {0}".format(quad_length))
        with self.gui_lock:
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

            rc = self.controller.get_parameter("scan", "roi_center")
            rd = self.controller.get_parameter("scan", "roi_dim")
            self.ui.image_raw_widget.roi.blockSignals(True)
            self.ui.image_raw_widget.roi.setPos((rc[0] - rd[0]/2, rc[1] - rd[1]/2))
            self.ui.image_raw_widget.roi.setSize((rd[0], rd[1]))
            self.ui.image_raw_widget.roi.show()
            self.ui.image_raw_widget.roi.blockSignals(False)
        self.update_image_selection()

    def update_image_threshold(self):
        """
        Set the threshold used when processing image for beam emittance calculations
        :return:
        """
        th = self.ui.threshold_spinbox.value()
        root.info("Setting image threshold to {0}".format(th))
        self.controller.set_parameter("analyse", "threshold", th)
        image_ind = self.ui.image_slider.value()
        k_ind = self.ui.k_slider.value()
        # self.controller.process_image(k_ind, image_ind)
        d = self.controller.process_all_images()

    def update_roi(self):
        """
        Update the roi in the raw image used for beam emittance calculations
        :return:
        """
        root.info("Update roi")
        rs = self.ui.image_raw_widget.roi.size()
        rp = self.ui.image_raw_widget.roi.pos()
        self.controller.set_parameter("scan", "roi_center", [rp[0] + rs[0]/2, rp[1] + rs[1]/2])
        self.controller.set_parameter("scan", "roi_dim", [rs[0], rs[1]])
        d = self.controller.process_all_images()
        # d.addCallback(self.update_image_selection)
        # d.addCallback(self.update_fit_data)

    def update_image_selection(self, result=None):
        """
        Update the image selected by the sliders.
        :return:
        """
        image_ind = self.ui.image_slider.value()
        k_ind = self.ui.k_slider.value()
        raw_data = self.controller.get_result("scan", "raw_data")
        proc_data = self.controller.get_result("scan", "proc_data")
        if raw_data is None:
            # Exit if there is no data
            return
        root.debug("Updating image to {0}:{1} in an array of {2}:{3}".format(k_ind, image_ind,
                                                                             len(raw_data), len(raw_data[0])))
        try:
            raw_pic = raw_data[k_ind][image_ind]
            proc_pic = proc_data[k_ind][image_ind]
            line_x = self.controller.get_result("scan", "line_data_x")[k_ind][image_ind]
            line_y = self.controller.get_result("scan", "line_data_y")[k_ind][image_ind]
        except IndexError:
            root.error("IndexError")
            raw_pic = np.random.random((64, 64))
            proc_pic = np.random.random((64, 64))
            line_x = np.random.random(64)
            line_y = np.random.random(64)
        self.ui.image_raw_widget.setImage(raw_pic)
        self.ui.image_proc_widget.setImage(proc_pic, scale=self.controller.get_parameter("scan", "pixel_size"))
        x_cent = [self.controller.get_result("scan", "x_cent")[k_ind][image_ind]]
        y_cent = [self.controller.get_result("scan", "y_cent")[k_ind][image_ind]]
        self.cent_plot.setData(y_cent, x_cent, symbol="x", symbolBrush="w")
        self.line_x_plot.setData(y=line_x)
        self.line_y_plot.setData(y=line_y)
        self.ui.image_raw_widget.roi.show()
        self.ui.image_proc_widget.updateImage()

    def update_fit_data(self, result=None):
        root.info("Updating fit data")
        k_data = np.array(self.controller.get_result("scan", "k_data")).flatten()
        sigma_data = np.array(self.controller.get_result("scan", "sigma_x")).flatten()
        en_data = np.array(self.controller.get_result("scan", "enabled_data")).flatten()
        symbol_list = list()
        brush_list = list()
        o_brush = pq.mkBrush(150, 150, 250, 150)
        x_brush = pq.mkBrush(250, 150, 150, 190)
        for en in en_data:
            if en == False:
                symbol_list.append("t")
                brush_list.append(x_brush)
            else:
                symbol_list.append("o")
                brush_list.append(o_brush)
        self.sigma_x_plot.setData(x=k_data, y=sigma_data**2, symbol=symbol_list, symbolBrush=brush_list, symbolPen=None,
                                  pen=None)
        poly = self.controller.get_result("scan", "fit_poly")
        x = np.linspace(k_data.min(), k_data.max(), 100)
        y = np.polyval(poly, x)
        self.fit_x_plot.setData(x=x, y=y)

    def points_clicked(self, scatterplotitem, point_list):
        try:
            pos = point_list[0].pos()
            root.info("Point clicked: {0}".format(pos))
        except IndexError:
            root.debug("No points in list - exit")
            return
        sx = self.controller.get_result("scan", "sigma_x")
        kd = self.controller.get_result("scan", "k_data")
        eps = 1e-9
        k_ind = None
        image_ind = None
        for k_i, k_list in enumerate(kd):
            for im_i, k_val in enumerate(k_list):
                if abs(pos.x() - k_val) < eps:
                    if abs(pos.y() - (sx[k_i][im_i])**2) < eps:
                        k_ind = k_i
                        image_ind = im_i
                        break
        if k_ind is not None:
            en_data = self.controller.get_result("scan", "enabled_data")
            enabled = en_data[k_ind][image_ind]
            en_data[k_ind][image_ind] = not en_data[k_ind][image_ind]
            if enabled is True:
                point_list[0].setSymbol("x")
                point_list[0].setBrush((200, 50, 50))
            else:
                point_list[0].setSymbol("o")
                point_list[0].setBrush((150, 150, 250, 100))
            self.ui.k_slider.blockSignals(True)
            self.ui.k_slider.setValue(k_ind)
            self.ui.image_slider.setValue(image_ind)
            self.ui.k_slider.blockSignals(False)

            self.update_image_selection()
            self.controller.fit_quad_data()
        else:
            root.debug("No point found")

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

        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    myapp.show()
    sys.exit(app.exec_())
