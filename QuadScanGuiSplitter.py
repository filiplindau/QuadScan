# -*- coding: utf-8 -*-
"""
Created 2018-12-17

Gui with splitters to set relative size of areas.

@author: Filip Lindau
"""

from PyQt4 import QtGui, QtCore

import pyqtgraph as pq
import sys
import numpy as np
import itertools
from QuadScanController import QuadScanController
from QuadScanState import StateDispatcher
from quadscan_gui_splitter import Ui_QuadScanDialog
import threading
import time
from QuadScanTasks import *
from QuadScanDataStructs import *

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


class MyScatterPlotItem(pq.ScatterPlotItem):
    sigRightClicked = QtCore.Signal(object, object, object)  ## self, points, right

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                self.sigClicked.emit(self, self.ptsClicked)
                ev.accept()
            else:
                ev.ignore()
        elif ev.button() == QtCore.Qt.RightButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsRightClicked = pts
                self.sigRightClicked.emit(self, self.ptsRightClicked, True)
                ev.accept()
            else:
                ev.ignore()
        else:
            ev.ignore()


class QuadScanGui(QtGui.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """

    load_done_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        root.debug("Init")
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'QuadScan')

        self.current_state = "unknown"
        self.last_load_dir = "."
        self.data_base_dir = "."
        self.section_init_flag = False
        self.screen_init_flag = True

        self.line_x_plot = None
        self.line_y_plot = None
        self.cent_plot = None
        self.sigma_x_plot = None
        self.fit_x_plot = None
        self.charge_plot = None
        self.fit_plot_vb = None

        self.camera_proxy = None    # Signal proxy to track mouse position over image
        self.process_image_proxy = None  # Signal proxy to track mouse position over image
        self.scan_proc_proxy = None  # Signal proxy to track mouse position over image

        self.data_store = DataStore()

        self.gui_lock = threading.Lock()

        self.ui = Ui_QuadScanDialog()
        self.ui.setupUi(self)

        self.setup_layout()

        self.image_processor = ImageProcessorTask(threshold=self.ui.p_threshold_spinbox.value(),
                                                  kernel=self.ui.p_median_kernel_spinbox.value(),
                                                  process_exec="process",
                                                  name="gui_image_proc")
        self.image_processor.start()
        # self.state_dispatcher = StateDispatcher(self.controller)
        # self.state_dispatcher.start()

        root.info("Exit gui init")

    def setup_layout(self):
        """
        Setup GUI layout and set stored settings
        :return:
        """
        # Plotting widgets:
        self.ui.camera_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.ui.camera_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ui.camera_widget.getView().setAspectLocked(False)
        self.ui.camera_widget.setImage(np.random.random((64, 64)))
        self.ui.camera_widget.ui.roiBtn.hide()
        self.ui.camera_widget.ui.menuBtn.hide()
        self.ui.camera_widget.roi.sigRegionChanged.disconnect()
        self.ui.camera_widget.roi.show()

        self.ui.camera_widget.roi.blockSignals(True)
        self.ui.camera_widget.roi.setPos((0, 0))
        self.ui.camera_widget.roi.setSize((64, 64))
        self.ui.camera_widget.roi.blockSignals(False)

        self.ui.process_image_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.ui.process_image_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ui.process_image_widget.getView().setAspectLocked(False)
        self.ui.process_image_widget.setImage(np.random.random((64, 64)))
        self.ui.process_image_widget.ui.roiBtn.hide()
        self.ui.process_image_widget.ui.menuBtn.hide()
        self.ui.process_image_widget.roi.sigRegionChanged.disconnect()
        h = self.ui.process_image_widget.getHistogramWidget()
        # h.item.sigLevelChangeFinished.connect(self.update_image_threshold)
        self.ui.process_image_widget.roi.show()

        self.ui.process_image_widget.roi.blockSignals(True)
        self.ui.process_image_widget.roi.setPos((0, 0))
        self.ui.process_image_widget.roi.setSize((64, 64))
        self.ui.process_image_widget.roi.blockSignals(False)

        # self.line_x_plot = self.ui.lineout_widget.plot()
        # self.line_x_plot.setPen((200, 25, 10))
        # self.line_y_plot = self.ui.lineout_widget.plot()
        # self.line_y_plot.setPen((10, 200, 25))
        # self.ui.lineout_widget.setLabel("bottom", "Line coord", "px")
        # self.ui.lineout_widget.showGrid(True)

        # self.sigma_x_plot = self.ui.fit_widget.plot()
        self.sigma_x_plot = MyScatterPlotItem()
        self.ui.fit_widget.getPlotItem().addItem(self.sigma_x_plot)
        self.sigma_x_plot.setPen((10, 200, 25))
        self.fit_x_plot = self.ui.fit_widget.plot()
        self.fit_x_plot.setPen(pq.mkPen(color=(180, 180, 250), width=2))
        self.ui.fit_widget.setLabel("bottom", "K", " 1/m²")
        self.ui.fit_widget.setLabel("left", "sigma", "m")
        self.ui.fit_widget.getPlotItem().showGrid(alpha=0.3)

        self.charge_plot = self.ui.charge_widget.plot()
        self.charge_plot.setPen((180, 250, 180))
        self.ui.charge_widget.setLabel("bottom", "K", " 1/m²")
        self.ui.charge_widget.setLabel("left", "charge", "a.u.")
        self.ui.charge_widget.getPlotItem().showGrid(alpha=0.3)
        self.ui.charge_widget.disableAutoRange()

        # Combobox init
        self.ui.fit_algo_combobox.addItem("Full matrix repr")
        self.ui.fit_algo_combobox.addItem("Thin lens approx")
        self.ui.fit_algo_combobox.setCurrentIndex(0)

        sections = ["MS1", "MS2", "MS3", "SP02"]
        for sect in sections:
            self.ui.section_combobox.addItem(sect.upper())

        doc = self.ui.status_textedit.document()
        doc.setMaximumBlockCount(100)

        # This is to make sure . is the decimal character
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))

        # Restore settings
        self.last_load_dir = self.settings.value("load_path", ".", type=str)
        self.data_base_dir = self.settings.value("base_path", ".", type=str)
        self.ui.save_path_linedit.setText(self.data_base_dir)
        val = self.settings.value("threshold", "0.0", type=float)
        self.ui.p_threshold_spinbox.setValue(val)
        val = self.settings.value("median_kernel", "3", type=int)
        self.ui.p_median_kernel_spinbox.setValue(val)

        val = self.settings.value("fit_algo", "thin_lens", type=str)
        if val == "thin_lens":
            ind = self.ui.fit_algo_combobox.findText("Thin lens approx")
        else:
            ind = self.ui.fit_algo_combobox.findText("Full matrix repr")
        root.debug("Fit algo index: {0}".format(ind))
        self.ui.fit_algo_combobox.setCurrentIndex(ind)

        k_start = self.settings.value("k_start", "0", type=float)
        self.ui.k_start_spinbox.setValue(k_start)

        k_end = self.settings.value("k_end", "0", type=float)
        self.ui.k_end_spinbox.setValue(k_end)

        val = str(self.settings.value("section", "ms1", type=str)).upper()
        ind = self.ui.section_combobox.findText(val)
        root.debug("Section {1} index: {0}".format(ind, val))
        self.ui.section_combobox.setCurrentIndex(ind)

        val = self.settings.value("num_k_values", "10", type=int)
        self.ui.num_k_spinbox.setValue(val)
        val = self.settings.value("num_shots", "2", type=int)
        self.ui.num_images_spinbox.setValue(val)

        # Signal connections
        self.ui.p_threshold_spinbox.editingFinished.connect(self.update_image_processing)
        self.ui.p_median_kernel_spinbox.editingFinished.connect(self.update_image_processing)
        self.ui.p_k_index_slider.valueChanged.connect(self.update_image_selection)
        self.ui.p_image_index_slider.valueChanged.connect(self.update_image_selection)
        self.sigma_x_plot.sigClicked.connect(self.points_clicked)
        self.sigma_x_plot.sigRightClicked.connect(self.points_clicked)
        self.ui.fit_algo_combobox.currentIndexChanged.connect(self.set_algo)
        self.ui.load_disk_button.clicked.connect(self.load_data)
        self.ui.set_start_k_button.clicked.connect(self.set_start_k)
        self.ui.set_end_k_button.clicked.connect(self.set_end_k)
        self.ui.k_current_spinbox.editingFinished.connect(self.set_current_k)
        self.ui.process_button.clicked.connect(self.start_processing)
        # self.ui.data_base_dir_button.clicked.connect(self.set_base_dir)
        self.ui.camera_start_button.clicked.connect(self.start_camera)
        self.ui.camera_stop_button.clicked.connect(self.stop_camera)
        self.ui.process_image_widget.roi.sigRegionChangeFinished.connect(self.update_process_image_roi)
        self.ui.camera_widget.roi.sigRegionChangeFinished.connect(self.update_camera_roi)
        self.ui.p_roi_cent_x_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_cent_y_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_w_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_h_spinbox.editingFinished.connect(self.set_roi)
        self.ui.scan_start_button.clicked.connect(self.start_scan)
        self.ui.scan_stop_button.clicked.connect(self.stop_scan)

        self.ui.section_combobox.currentIndexChanged.connect(self.update_section)
        self.ui.quad_combobox.currentIndexChanged.connect(self.update_section)
        self.ui.screen_combobox.currentIndexChanged.connect(self.update_section)

        # self.controller.image_done_signal.connect(self.update_fit_data)

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

        # Install event filter
        self.ui.k_current_spinbox.installEventFilter(self)

        # Setup signal proxies for mouse tracking
        self.camera_proxy = pq.SignalProxy(self.ui.camera_widget.scene.sigMouseMoved,
                                           rateLimit=30, slot=self.camera_mouse_moved)
        self.process_image_proxy = pq.SignalProxy(self.ui.process_image_widget.scene.sigMouseMoved,
                                                  rateLimit=30, slot=self.process_image_mouse_moved)

    def eventFilter(self, obj, event):
        """
        Used for intercepting wheel events to modify magnet k-value
        :param obj:
        :param event:
        :return:
        """
        if event.type() == QtCore.QEvent.Wheel:
            dk = 0.025 * event.delta() / 120.0
            self.ui.k_current_spinbox.setValue(self.ui.k_current_spinbox.value() + dk)
            return True
        else:
            return False

    def closeEvent(self, event):
        """
        Closing the applications. Stopping threads and saving the settings.
        :param event:
        :return:
        """
        self.image_processor.stop_processing()
        self.settings.setValue("load_path", self.last_load_dir)
        self.settings.setValue("base_path", self.data_base_dir)

        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))

        self.settings.setValue("threshold", self.ui.p_threshold_spinbox.value())
        # self.settings.setValue("median_kernel", self.controller.get_parameter("analysis", "median_kernel"))
        # self.settings.setValue("fit_algo", self.controller.get_parameter("analysis", "fit_algo"))
        # self.settings.setValue("k_start", self.ui.k_start_spinbox.value())
        # self.settings.setValue("k_end", self.ui.k_end_spinbox.value())
        # self.settings.setValue("num_shots", self.controller.get_parameter("scan", "num_shots"))
        # self.settings.setValue("num_k_values", self.controller.get_parameter("scan", "num_k_values"))

        # self.settings.setValue("section", self.controller.get_parameter("scan", "section_name"))
        # self.settings.setValue("section_quad", self.controller.get_parameter("scan", "quad_name"))
        # self.settings.setValue("section_screen", self.controller.get_parameter("scan", "screen_name"))

    def set_roi(self):
        root.info("Set roi from spinboxes")

    def load_data(self):
        """
        Initiate load data state
        :return:
        """
        root.info("Loading data from disk")
        load_dir = QtGui.QFileDialog.getExistingDirectory(self, "Select directory", self.last_load_dir)
        self.last_load_dir = load_dir
        root.debug("Loading from directory {0}".format(load_dir))
        self.image_processor.add_callback(self.update_load_data)
        t1 = LoadQuadScanDirTask(str(load_dir), process_now=True,
                                 threshold=self.ui.p_threshold_spinbox.value(),
                                 kernel_size=self.ui.p_median_kernel_spinbox.value(),
                                 image_processor_task=self.image_processor,
                                 process_exec_type="process",
                                 name="load_task", callback_list=[self.update_load_data])
        t1.start()
        source_name = QtCore.QDir.fromNativeSeparators(load_dir).split("/")[-1]
        self.ui.data_source_label.setText(source_name)

    def update_load_data(self, task):
        root.info("Update load data {0}".format(task.name))
        if task is not None:
            if task.is_done() is False:
                image = task.get_result(wait=False)   # type: ProcessedImage
                # root.debug("image {0}".format(image.pic_roi))
                self.update_image_selection(image.pic_roi)
            else:
                task.remove_callback(self.update_load_data)
                quad_scan_data = task.get_result(wait=False)   # type: QuadScanData
                self.data_store.quad_scan_data = quad_scan_data
                self.update_analysis_parameters()

    def update_analysis_parameters(self):
        acc_params = self.data_store.quad_scan_data.acc_params  # type: AcceleratorParameters
        self.ui.p_electron_energy_label.setText(acc_params.electron_energy)
        self.ui.p_quad_length_label.setText(acc_params.quad_length)
        self.ui.p_quad_screen_dist_label.setText(acc_params.quad_screen_dist)

        self.ui.p_k_index_slider.setMaximum(acc_params.num_k)
        self.ui.p_image_index_slider.setMaximum(acc_params.num_images)

    def update_section(self):
        root.info("Updating section")

    def update_scan_devices(self):
        root.info("Updating scan devices")

    def update_process_image_roi(self):
        root.info("Updating ROI for process image")

    def update_camera_roi(self):
        root.info("Updating ROI for camera image")

    def update_image_processing(self):
        root.info("Processing images")

    def update_image_selection(self, image=None):
        root.info("Updating image ")
        if image is None:
            k_ind = self.ui.p_k_index_slider.value()
            im_ind = self.ui.p_image_index_slider.value()
            image = self.data_store.quad_scan_data.proc_images[k_ind][im_ind].pic_roi
        self.ui.process_image_widget.setImage(image)

    def set_algo(self):
        root.info("Setting fit algo")

    def set_start_k(self):
        root.info("Setting start k value to {0}".format(self.ui.k_start_spinbox.value()))

    def set_end_k(self):
        root.info("Setting end k value to {0}".format(self.ui.k_end_spinbox.value()))

    def set_current_k(self):
        root.info("Setting current k to {0}".format(self.ui.k_current_spinbox))
        self.ui.current_k_label.setText("k = {0:.3f} 1/m²".format(self.ui.k_current_spinbox.value()))

    def set_base_dir(self):
        root.info("Setting base save directory")

    def start_camera(self):
        root.info("Start camera pressed")

    def stop_camera(self):
        root.info("Stop camera pressed")

    def start_scan(self):
        root.info("Start scan pressed")

    def stop_scan(self):
        root.info("Stop scan pressed")

    def start_processing(self):
        root.info("Start processing")

    def points_clicked(self, scatterplotitem, point_list, right=False):
        """
        Check if there is a point in the clicked list that should be enabled or disabled.
        Right click disabled, left click enables.

        :param scatterplotitem: Scatterplot that was clicked
        :param point_list: List of points under the mouse cursor
        :param right: True if the right mouse button was clicked, False if left.
        :return:
        """
        try:
            pos = point_list[0].pos()
            root.info("Point clicked: {0}".format(pos))
        except IndexError:
            root.debug("No points in list - exit")
            return
        root.debug("Right button: {0}".format(right))
        # sx = self.controller.get_result("scan", "sigma_x")
        # kd = self.controller.get_result("scan", "k_data")
        # en_data = self.controller.get_result("scan", "enabled_data")
        sx = []
        kd = []
        en_data = []
        enabled = not right
        eps = 1e-9
        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        root.debug("Mouse pos: {0}".format(mouse_pos))
        k_sel_i = None
        im_sel_i = None
        sel_dist = np.inf
        k_tog_i = None
        im_tog_i = None
        tog_dist = np.inf
        tog_spot = None
        for p in point_list:
            pos = p.pos()
            # We need to loop through the list of data points to find the index of the points clicked:
            for k_i, k_list in enumerate(kd):
                for im_i, k_val in enumerate(k_list):
                    # Check if the point is within eps:
                    if abs(pos.x() - k_val) < eps:
                        if abs(pos.y() - (sx[k_i][im_i])) < eps:
                            d = (pos.x() - mouse_pos.x())**2 + (pos.y() - mouse_pos.y())**2
                            if d < sel_dist:
                                k_sel_i = k_i
                                im_sel_i = im_i
                                sel_dist = d
                            if en_data[k_i][im_i] != enabled:
                                if d < tog_dist:
                                    k_tog_i = k_i
                                    im_tog_i = im_i
                                    tog_dist = d
                                    tog_spot = p

        if k_tog_i is not None:
            en_data[k_tog_i][im_tog_i] = enabled

            if enabled is False:
                tog_spot.setSymbol("x")
                tog_spot.setBrush((200, 50, 50))
            else:
                tog_spot.setSymbol("o")
                tog_spot.setBrush((150, 150, 250, 100))
        if k_sel_i is not None:
            # self.ui.k_slider.blockSignals(True)
            # self.ui.k_slider.setValue(k_sel_i)
            # self.ui.image_slider.setValue(im_sel_i)
            # self.ui.k_slider.blockSignals(False)

            self.update_image_selection()
            # self.controller.fit_quad_data()

    def camera_mouse_moved(self, event):
        pos = self.ui.camera_widget.view.mapSceneToView(event[0])
        pic = self.ui.camera_widget.getProcessedImage()
        x = int(pos.x())
        y = int(pos.y())
        if x >= 0 and y >= 0:
            try:
                intensity = pic[x, y]
            except IndexError:
                return
            self.ui.mouse_label.setText(
                "Cam image at ({0}, {1}) px: {2:.0f}".format(min(x, 9999), min(y, 9999), intensity))

    def process_image_mouse_moved(self, event):
        pos = self.ui.process_image_widget.view.mapSceneToView(event[0])
        pic = self.ui.process_image_widget.getProcessedImage()
        x = int(pos.x())
        y = int(pos.y())
        if x >= 0 and y >= 0:
            try:
                intensity = pic[x, y]
            except IndexError:
                return
            self.ui.mouse_label.setText(
                "Proc image at ({0}, {1}) px: {2:.0f}".format(min(x, 9999), min(y, 9999), intensity))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    root.info("QuadScanGui object created")
    myapp.show()
    root.info("App show")
    sys.exit(app.exec_())
    root.info("App exit")
