# -*- coding: utf-8 -*-
"""
Created 2018-12-17

Gui with splitters to set relative size of areas.

@author: Filip Lindau
"""

from PyQt4 import QtGui, QtCore

import pyqtgraph as pq
import sys
import glob
import numpy as np
import itertools
from quadscan_gui_splitter import Ui_QuadScanDialog
from scandata_file_dialog import OpenScanFileDialog
from collections import OrderedDict
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


no_database = False
dummy_name_dict = {"mag": "192.168.1.101:10000/i-ms1/mag/qb-01#dbase=no",
                   "crq": "192.168.1.101:10000/i-ms1/mag/qb-01#dbase=no",
                   "screen": "192.168.1.101:10001/i-ms1/dia/scrn-01#dbase=no",
                   "beamviewer": "192.168.1.101:10002/lima/beamviewer/i-ms1-dia-scrn-01#dbase=no",
                   "liveviewer": "192.168.1.101:10003/lima/liveviewer/i-ms1-dia-scrn-01#dbase=no",
                   "limaccd": "192.168.1.101:10004/lima/limaccd/i-ms1-dia-scrn-01#dbase=no"}


class MyScatterPlotItem(pq.ScatterPlotItem):
    """
    Subclassed to allow capture of right clicks and emitting signal.
    """
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


class MyHistogramItem(pq.HistogramLUTItem):
    def imageChanged(self, autoLevel=False, autoRange=False):
        root.info("Hist item!!!")
        # return pq.HistogramLUTItem.imageChanged(self, autoLevel, autoRange)

        if self.imageItem() is None:
            return

        h0 = self.imageItem().getHistogram()
        if h0[0] is None:
            return
        # h = (h0[0], (h0[0] * h0[1]).cumsum())
        h = (h0[0], (h0[0] * h0[1]))
        self.plot.setData(*h)
        if autoLevel:
            mn = h[0]
            mx = h[-1]
            self.region.setRegion([mn, mx])


class QuadScanGui(QtGui.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """

    load_done_signal = QtCore.Signal(object)
    update_fit_signal = QtCore.Signal()
    update_camera_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        root.debug("Init")
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'QuadScan')

        self.current_state = "unknown"
        self.last_load_dir = "."
        self.data_base_dir = "."
        self.scan_save_path = "."
        self.section_init_flag = True   # Set when selecting a new section in combobox
        self.screen_init_flag = True    # Set when selecting new screen in combobox
        self.quad_init_flag = True      # Set when selecting new quad in combobox
        self.load_init_flag = False     # Set when staring new load from disk

        self.line_x_plot = None
        self.line_y_plot = None
        self.cent_plot = None
        self.sigma_x_plot = None
        self.fit_x_plot = None
        self.charge_plot = None
        self.fit_plot_vb = None
        self.process_image_view = None       # ROI for when viewing raw process image
        self.load_image_max = 0.0
        self.scan_image_max = 0.0
        self.user_enable_list = list()

        self.camera_proxy = None    # Signal proxy to track mouse position over image
        self.process_image_proxy = None  # Signal proxy to track mouse position over image
        self.scan_proc_proxy = None  # Signal proxy to track mouse position over image

        self.quad_scan_data_analysis = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.quad_scan_data_scan = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.fit_result = FitResult(poly=None, alpha=None, beta=None, eps=None, eps_n=None,
                                    gamma_e=None, fit_data=None, residual=None)
        self.section_devices = SectionDevices(sect_quad_dict=None, sect_screen_dict=None)
        if no_database:
            self.device_handler = DeviceHandler(name="Handler")
        else:
            self.device_handler = DeviceHandler("g-v-csdb-0:10000", name="Handler")

        self.section_list = ["MS1", "MS2", "MS3", "SP02"]
        self.current_section = "MS1"
        self.current_quad = None        # type: SectionQuad
        self.current_screen = None      # type: SectionScreen
        self.camera_cal = [1, 1]          # type: list
        self.quad_tasks = list()        # Repeat tasks for selected quad
        self.screen_tasks = list()      # Repeat tasks for selected screen
        self.processing_tasks = list()
        self.scan_task = None

        self.gui_lock = threading.Lock()

        self.ui = Ui_QuadScanDialog()
        self.ui.setupUi(self)

        self.setup_layout()

        # self.image_processor = ImageProcessorTask(threshold=self.ui.p_threshold_spinbox.value(),
        #                                           kernel=self.ui.p_median_kernel_spinbox.value(),
        #                                           process_exec="process",
        #                                           name="gui_image_proc")
        # self.image_processor = ImageProcessorTask2(image_size=(1300, 2000), threshold=self.ui.p_threshold_spinbox.value(),
        #                                            kernel=self.ui.p_median_kernel_spinbox.value(),
        #                                            name="gui_image_proc")
        self.image_processor = ProcessAllImagesTask2(image_size=[2000, 2000], name="gui_image_proc",
                                                     callback_list=[self.update_image_processing])
        self.image_processor.start()

        if no_database:
            t1 = PopulateDummyDeviceList(sections=self.section_list, dummy_name_dict=dummy_name_dict,
                                         name="pop_sections")
        else:
            t1 = PopulateDeviceListTask(sections=self.section_list, name="pop_sections")
        t1.start()
        t1.add_callback(self.populate_sections)

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

        # hw = self.ui.process_image_widget.getHistogramWidget()
        # hw.item = MyHistogramItem()
        # hw.setCentralItem(hw.item)
        # hw.item.setImageItem(self.ui.process_image_widget.getImageItem())
        self.ui.process_image_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.ui.process_image_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ui.process_image_widget.getView().setAspectLocked(False)
        self.ui.process_image_widget.setImage(np.random.random((64, 64)))
        self.ui.process_image_widget.ui.roiBtn.hide()
        self.ui.process_image_widget.ui.menuBtn.hide()
        self.ui.process_image_widget.roi.sigRegionChanged.disconnect()
        h = self.ui.process_image_widget.getHistogramWidget()
        h.item.sigLevelChangeFinished.connect(self.update_process_image_threshold)
        self.ui.process_image_widget.roi.show()

        self.ui.process_image_widget.roi.blockSignals(True)
        roi = self.ui.process_image_widget.roi      # type: pq.ROI
        roi_handles = roi.getHandles()
        roi.removeHandle(roi_handles[1])
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

        # self.charge_plot = self.ui.charge_widget.plot()
        self.charge_plot = MyScatterPlotItem()
        self.ui.charge_widget.getPlotItem().addItem(self.charge_plot)
        self.charge_plot.setPen((180, 250, 180))
        self.ui.charge_widget.setLabel("bottom", "K", " 1/m²")
        self.ui.charge_widget.setLabel("left", "charge", "a.u.")
        self.ui.charge_widget.getPlotItem().showGrid(alpha=0.3)
        # self.ui.charge_widget.disableAutoRange()

        # Combobox init
        self.ui.fit_algo_combobox.addItem("Full matrix repr")
        self.ui.fit_algo_combobox.addItem("Thin lens approx")
        self.ui.fit_algo_combobox.setCurrentIndex(0)

        # Scan status init
        self.ui.scan_status_label.setText("STOPPED: k -/- image -/-")
        self.ui.scan_progress_label.setText("[----------]")

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

        val = self.settings.value("filtered_image_show", True, type=bool)
        if bool(val) is True:
            self.ui.p_filtered_image_radio.setChecked(True)
        else:
            self.ui.p_raw_image_radio.setChecked(True)

        val = self.settings.value("use_x_axis", True, type=bool)
        if bool(val) is True:
            self.ui.p_x_radio.setChecked(True)
        else:
            self.ui.p_y_radio.setChecked(True)

        k_start = self.settings.value("k_start", "0", type=float)
        self.ui.k_start_spinbox.setValue(k_start)

        k_end = self.settings.value("k_end", "0", type=float)
        self.ui.k_end_spinbox.setValue(k_end)

        val = str(self.settings.value("section", "ms1", type=str)).upper()
        ind = self.ui.section_combobox.findText(val)
        root.debug("Section {1} index: {0}".format(ind, val))
        self.ui.section_combobox.setCurrentIndex(ind)

        val = self.settings.value("num_k", "10", type=int)
        self.ui.num_k_spinbox.setValue(val)
        val = self.settings.value("num_images", "2", type=int)
        self.ui.num_images_spinbox.setValue(val)

        val = self.settings.value("electron_energy", "200", type=float)
        self.ui.electron_energy_spinbox.setValue(val)
        val = self.settings.value("rep_ate", "2", type=float)
        self.ui.reprate_spinbox.setValue(val)

        # Signal connections
        self.ui.set_start_k_button.clicked.connect(self.set_start_k)
        self.ui.set_end_k_button.clicked.connect(self.set_end_k)
        self.ui.k_current_spinbox.editingFinished.connect(self.set_current_k)
        # self.ui.data_base_dir_button.clicked.connect(self.set_base_dir)
        self.ui.camera_start_button.clicked.connect(self.start_camera)
        self.ui.camera_stop_button.clicked.connect(self.stop_camera)
        self.ui.camera_widget.roi.sigRegionChangeFinished.connect(self.update_camera_roi)
        self.ui.screen_in_button.clicked.connect(self.insert_screen)
        self.ui.screen_out_button.clicked.connect(self.remove_screen)
        self.ui.scan_start_button.clicked.connect(self.start_scan)
        self.ui.scan_stop_button.clicked.connect(self.stop_scan)

        self.ui.section_combobox.currentIndexChanged.connect(self.update_section)
        self.ui.quad_combobox.currentIndexChanged.connect(self.update_section)
        self.ui.screen_combobox.currentIndexChanged.connect(self.update_section)

        self.ui.process_image_widget.roi.sigRegionChangeFinished.connect(self.update_process_image_roi)
        hw = self.ui.process_image_widget.getHistogramWidget()
        # hw.sigLevelChangeFinished.connect(self.update_process_image_histogram)
        hw.item.blockSignals(True)
        self.ui.process_button.clicked.connect(self.start_processing)
        self.ui.p_threshold_spinbox.editingFinished.connect(self.start_processing)
        # self.ui.p_load_hist_button.clicked.connect(self.update_process_image_threshold)
        self.ui.p_median_kernel_spinbox.editingFinished.connect(self.start_processing)
        self.ui.p_image_index_slider.valueChanged.connect(self.update_image_selection)
        self.ui.p_raw_image_radio.toggled.connect(self.change_raw_or_processed_view)
        self.ui.p_x_radio.toggled.connect(self.change_analysis_axis)
        self.ui.p_enable_all_button.clicked.connect(self.enable_all_points)
        self.sigma_x_plot.sigClicked.connect(self.points_clicked)
        self.sigma_x_plot.sigRightClicked.connect(self.points_clicked)
        self.charge_plot.sigClicked.connect(self.points_clicked)
        self.charge_plot.sigRightClicked.connect(self.points_clicked)
        self.ui.fit_algo_combobox.currentIndexChanged.connect(self.set_algo)
        self.ui.load_disk_button.clicked.connect(self.load_data_disk)
        self.ui.load_scan_button.clicked.connect(self.load_data_scan)
        self.ui.p_roi_cent_x_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_cent_y_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_w_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_h_spinbox.editingFinished.connect(self.set_roi)

        self.update_fit_signal.connect(self.plot_sigma_data)
        self.update_camera_signal.connect(self.update_camera_image)

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

        scan_analysis_splitter_sizes = self.settings.value("scan_analysis_splitter", [None], type="QVariantList")
        if scan_analysis_splitter_sizes[0] is not None:
            self.ui.scan_analysis_splitter.setSizes([np.int(s) for s in scan_analysis_splitter_sizes])

        analysis_pic_plot_splitter_sizes = self.settings.value("analysis_pic_plot_splitter", [None], type="QVariantList")
        if analysis_pic_plot_splitter_sizes[0] is not None:
            self.ui.analysis_pic_plot_splitter.setSizes([np.int(s) for s in analysis_pic_plot_splitter_sizes])

        analysis_plots_splitter_sizes = self.settings.value("analysis_plots_splitter", [None], type="QVariantList")
        if analysis_plots_splitter_sizes[0] is not None:
            self.ui.analysis_plots_splitter.setSizes([np.int(s) for s in analysis_plots_splitter_sizes])

        tab_index = self.settings.value("tab_index", 0, type=int)
        self.ui.tabWidget.setCurrentIndex(tab_index)

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
        self.image_processor.clear_callback_list()
        root.debug("Stop image processor")
        self.image_processor.finish_processing()
        root.debug("Command sent.")
        for t in self.screen_tasks:
            try:
                root.debug("Cancelling {0}".format(t.get_name()))
                t.cancel()
            except AttributeError:
                pass
        for t in self.quad_tasks:
            try:
                root.debug("Cancelling {0}".format(t.get_name()))
                t.cancel()
            except AttributeError:
                pass
        self.settings.setValue("load_path", self.last_load_dir)
        self.settings.setValue("base_path", self.ui.save_path_linedit.text())

        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))

        self.settings.setValue("scan_analysis_splitter", self.ui.scan_analysis_splitter.sizes())
        self.settings.setValue("analysis_pic_plot_splitter", self.ui.analysis_pic_plot_splitter.sizes())
        self.settings.setValue("analysis_plots_splitter", self.ui.analysis_plots_splitter.sizes())
        self.settings.setValue("tab_index", self.ui.tabWidget.currentIndex())

        self.settings.setValue("threshold", self.ui.p_threshold_spinbox.value())
        self.settings.setValue("median_kernel", self.ui.p_median_kernel_spinbox.value())
        self.settings.setValue("filtered_image_show", self.ui.p_filtered_image_radio.isChecked())
        self.settings.setValue("use_x_axis", self.ui.p_x_radio.isChecked())

        if "Full matrix" in str(self.ui.fit_algo_combobox.currentText()):
            algo = "full matrix"
        else:
            algo = "thin lens"
        self.settings.setValue("fit_algo", algo)
        self.settings.setValue("k_start", self.ui.k_start_spinbox.value())
        self.settings.setValue("k_end", self.ui.k_end_spinbox.value())
        self.settings.setValue("num_images", self.ui.num_images_spinbox.value())
        self.settings.setValue("num_k", self.ui.num_k_spinbox.value())
        self.settings.setValue("electron_energy", self.ui.electron_energy_spinbox.value())
        self.settings.setValue("rep_rate", self.ui.reprate_spinbox.value())

        self.settings.setValue("section", self.ui.section_combobox.currentText())
        self.settings.setValue("section_quad", self.ui.quad_combobox.currentText())
        self.settings.setValue("section_screen", self.ui.screen_combobox.currentText())
        root.debug("Settings done.")

    def set_roi(self):
        root.info("Set roi from spinboxes")
        roi_x = self.ui.p_roi_cent_x_spinbox.value()
        roi_y = self.ui.p_roi_cent_y_spinbox.value()
        roi_w = self.ui.p_roi_size_w_spinbox.value()
        roi_h = self.ui.p_roi_size_h_spinbox.value()
        pos = [roi_x - roi_w / 2.0, roi_y - roi_h / 2.0]

        self.ui.process_image_widget.roi.blockSignals(True)
        self.ui.process_image_widget.roi.setPos(pos, update=False)
        self.ui.process_image_widget.roi.setSize([roi_w, roi_h])
        self.ui.process_image_widget.roi.blockSignals(False)

        self.ui.process_image_widget
        self.start_processing()

    def load_data_disk(self):
        """
        Initiate load data from save directory. Starts a LoadQuadScanTask and sets a callback update_load_data
        when completed.

        :return:
        """
        root.info("Loading data from disk")
        filedialog = OpenScanFileDialog(self.last_load_dir)
        res = filedialog.exec_()
        root.debug("Load dir return value: {0}".format(res))
        if res != QtGui.QDialog.Accepted:
            return
        load_dir = filedialog.get_selected_path()
        self.last_load_dir = load_dir
        root.debug("Loading from directory {0}".format(load_dir))
        self.ui.process_image_widget.getHistogramWidget().item.blockSignals(True)    # Block signals to avoid threshold problems
        self.load_image_max = 0.0

        self.load_init_flag = True

        # LoadQuadScanTask takes care of the actual loading of the files in the specified directory:
        t1 = LoadQuadScanDirTask(str(load_dir), process_now=True,
                                 threshold=self.ui.p_threshold_spinbox.value(),
                                 kernel_size=self.ui.p_median_kernel_spinbox.value(),
                                 process_exec_type="thread",
                                 name="load_task", callback_list=[self.update_load_data])
        t1.start()
        source_name = QtCore.QDir.fromNativeSeparators(load_dir).split("/")[-1]
        self.ui.data_source_label.setText(source_name)

    def load_data_scan(self):
        """
        Move data from scan to analysis datastructures.

        :return:
        """
        root.info("Loading data from scan")
        if self.quad_scan_data_scan.acc_params is not None:
            root.debug("Load data complete. Storing quad scan data.")
            source_name = "Scan data {0}-{1}".format(self.current_quad.mag, self.current_screen.screen)
            self.ui.data_source_label.setText(source_name)

            hw = self.ui.process_image_widget.getHistogramWidget()
            hl = hw.getLevels()
            hw.setLevels(self.ui.p_threshold_spinbox.value(), self.scan_image_max)
            root.debug("Proc images len: {0}".format(len(self.quad_scan_data_scan.proc_images)))
            self.user_enable_list = [True for x in range(len(self.quad_scan_data_scan.proc_images))]
            self.update_analysis_parameters()
            self.update_image_selection()

            self.quad_scan_data_analysis = self.quad_scan_data_scan
            root.debug("Slider max {0}".format(len(self.quad_scan_data_scan.images)-1))
            self.ui.p_image_index_slider.setMaximum(len(self.quad_scan_data_scan.images)-1)
            self.start_processing()
            # self.update_analysis_parameters()
            # self.update_image_selection()
            # self.update_fit_signal.emit()
            # self.start_fit()
            root.debug("Quad_scan_data_analysis {0}, "
                       "quad_scan_data_scan {1}".format(len(self.quad_scan_data_analysis.proc_images),
                                                        len(self.quad_scan_data_scan.proc_images)))

    def load_dir_entered(self, load_dir):
        s_dir = str(load_dir)
        file_list = glob.glob("{0}/*.png".format(s_dir))
        root.info("Entered directory {0}. {1} files found.".format(s_dir, len(file_list)))
        filename = "daq_info.txt"
        if os.path.isfile(os.path.join(s_dir, filename)) is False:
            s = "daq_info.txt not found"
        else:
            root.debug("Loading Jason format data")
            data_dict = dict()
            with open(os.path.join(s_dir, filename), "r") as daq_file:
                while True:
                    line = daq_file.readline()
                    if line == "" or line[0:5] == "*****":
                        break
                    try:
                        key, value = line.split(":")
                        data_dict[key.strip()] = value.strip()
                    except ValueError:
                        pass
            try:
                s = "Load data. {0} images, {1}, {2} < k < {3}, {4} MeV".format(len(file_list),
                                                                                data_dict["quad"],
                                                                                data_dict["k_min"],
                                                                                data_dict["k_max"],
                                                                                data_dict["beam_energy"])
            except KeyError:
                s = "Load data. {0} images. Could not parse daq_info.txt".format(len(file_list))
        self.sender().setWindowTitle(s)

    def update_load_data(self, task):
        """
        Callback function for loading data from disk.
        It can be called during the load for each processed image and
        finally when completed.

        For each image: Update image selection
        When completed: Store quad scan data and start fit

        :param task:
        :return:
        """
        root.info("Update load data {0}, {1}".format(task.name, self.load_init_flag))
        if task is not None:
            if task.is_done() is False:
                # Task is not done so this is an image update
                image = task.get_result(wait=False)   # type: QuadImage
                acc_params = task.acc_params
                if task.is_cancelled():
                    root.error("Error when loading image: {0}".format(image))
                else:
                    m = image.image.max()
                    if m > self.load_image_max:
                        self.load_image_max = m
                    # root.debug("image {0}".format(image.pic_roi))

                    if self.load_init_flag:
                        pos = [acc_params.roi_center[1] - acc_params.roi_dim[1] / 2.0,
                               acc_params.roi_center[0] - acc_params.roi_dim[0] / 2.0]

                        self.process_image_view = [0, 0, acc_params.roi_dim[1], acc_params.roi_dim[0]]
                        x_range = [pos[0], pos[0] + self.process_image_view[2]]
                        y_range = [pos[1], pos[1] + self.process_image_view[3]]
                        root.debug("Init image view {0}, {1}".format(x_range, y_range))
                        self.ui.process_image_widget.view.setAspectLocked(True, 1)
                        self.ui.process_image_widget.view.setRange(xRange=x_range, yRange=y_range)
                        self.load_init_flag = False

                    hw = self.ui.process_image_widget.getHistogramWidget()  # type: pq.HistogramLUTWidget
                    hw.item.blockSignals(True)
                    self.update_image_selection(image.image, auto_levels=True, auto_range=False)
            else:
                root.debug("Load data complete. Storing quad scan data.")
                hw = self.ui.process_image_widget.getHistogramWidget()      # type: pq.HistogramLUTWidget
                hw.item.blockSignals(True)
                self.ui.p_threshold_spinbox.blockSignals(True)
                hl = hw.getLevels()
                hw.setLevels(self.ui.p_threshold_spinbox.value(), self.load_image_max)
                hw.item.blockSignals(False)
                self.ui.p_threshold_spinbox.blockSignals(False)
                task.remove_callback(self.update_load_data)
                if isinstance(task, LoadQuadScanDirTask):
                    result = task.get_result(wait=False)   # type: QuadScanData
                    if task.is_cancelled():
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                        msg = "Load dir error: {0}".format(result)
                        self.ui.status_textedit.append("\n---------------------------\n"
                                                       "{0}:\n"
                                                       "{1}\n".format(time_str, msg))

                        root.error(msg)
                    else:
                        self.quad_scan_data_analysis = result
                        root.debug("Acc parameters: {0}".format(result.acc_params))
                        self.ui.p_image_index_slider.setMaximum(len(self.quad_scan_data_analysis.images) - 1)
                        self.ui.p_image_index_slider.setValue(0)
                        self.ui.p_image_index_slider.update()
                        root.debug("Proc images len: {0}".format(len(result.proc_images)))
                        root.debug("Images len: {0}".format(len(result.images)))
                        self.user_enable_list = [True for x in range(len(result.proc_images))]
                        self.update_analysis_parameters()
                        self.update_image_selection()
                        self.start_processing()
                        # self.update_fit_signal.emit()
                        # self.start_fit()

    def update_analysis_parameters(self):
        root.debug("Acc params {0}".format(self.quad_scan_data_analysis.acc_params))
        acc_params = self.quad_scan_data_analysis.acc_params  # type: AcceleratorParameters
        self.ui.p_electron_energy_label.setText("{0:.2f} MeV".format(acc_params.electron_energy))
        self.ui.p_quad_length_label.setText("{0:.2f} m".format(acc_params.quad_length))
        self.ui.p_quad_screen_dist_label.setText("{0:.2f} m".format(acc_params.quad_screen_dist))

        self.ui.p_roi_cent_x_spinbox.setValue(acc_params.roi_center[1])
        self.ui.p_roi_cent_y_spinbox.setValue(acc_params.roi_center[0])
        self.ui.p_roi_size_w_spinbox.setValue(acc_params.roi_dim[1])
        self.ui.p_roi_size_h_spinbox.setValue(acc_params.roi_dim[0])
        # Init image view as the ROI:
        pos = [acc_params.roi_center[1] - acc_params.roi_dim[1] / 2.0,
               acc_params.roi_center[0] - acc_params.roi_dim[0] / 2.0]
        if self.ui.p_raw_image_radio.isChecked():
            self.process_image_view = [0, 0, acc_params.roi_dim[1], acc_params.roi_dim[0]]
            x_range = [pos[0], pos[0] + self.process_image_view[2]]
            y_range = [pos[1], pos[1] + self.process_image_view[3]]
        else:
            self.process_image_view = [pos[0], pos[1], acc_params.roi_dim[1], acc_params.roi_dim[0]]
            x_range = [0, self.process_image_view[2]]
            y_range = [0, self.process_image_view[3]]
        root.debug("x range: {0}, y range: {1}".format(x_range, y_range))
        self.ui.process_image_widget.view.setRange(xRange=x_range, yRange=y_range)

        self.ui.process_image_widget.roi.blockSignals(True)
        self.ui.process_image_widget.roi.setPos(pos, update=False)
        self.ui.process_image_widget.roi.setSize([acc_params.roi_dim[1], acc_params.roi_dim[0]])
        self.ui.process_image_widget.roi.blockSignals(False)

        # self.ui.p_image_index_slider.setMaximum(acc_params.num_images-1)
        self.ui.p_image_index_slider.setMaximum(len(self.quad_scan_data_analysis.proc_images) - 1)
        th_list = [i.threshold for i in self.quad_scan_data_analysis.proc_images]
        try:
            threshold = sum(th_list) * 1.0 / len(self.quad_scan_data_analysis.proc_images)
            self.ui.p_threshold_spinbox.setValue(threshold)
        except ZeroDivisionError:
            threshold = 0.0
        root.debug("Setting threshold to {0}".format(threshold))

    def update_section(self):
        """
        Update the section according current selections. Checks if a new section has been chosen
        or if a new magnet/screen within the current section has been chosen.
        :return:
        """
        sect = str(self.ui.section_combobox.currentText()).upper()
        root.info("Update section settings to {0}".format(sect))
        try:
            quads = self.section_devices.sect_quad_dict[sect]
            screens = self.section_devices.sect_screen_dict[sect]
        except KeyError as e:
            # Section not in dict. Exit
            root.exception("Section {0} not in dict {1}".format(sect, self.section_devices.sect_quad_dict.keys()))
            return

        # Check if a new section was chosen, then re-populate the comboboxes for magnets and screens
        self.ui.quad_combobox.blockSignals(True)
        self.ui.screen_combobox.blockSignals(True)
        if sect != self.current_section or self.section_init_flag is True:
            root.debug("New section, populating comboboxes")
            root.debug("Number of quads: {0}".format(len(quads)))
            self.ui.quad_combobox.clear()
            self.ui.screen_combobox.clear()

            # root.debug("Quad combobox count: {0}".format(self.ui.quad_combobox.count()))
            for qd in quads:
                self.ui.quad_combobox.addItem(qd.name.upper())
            for sc in screens:
                self.ui.screen_combobox.addItem(sc.name.upper())
            try:
                self.ui.quad_combobox.setCurrentIndex(0)
                self.ui.screen_combobox.setCurrentIndex(0)
            except IndexError:
                # Quad, screen lists not populated. Cannot select device yet
                return
            if self.section_init_flag:
                # If this is the initial call to update_section we set the selection to previous values:
                val = str(self.settings.value("section_quad", "NA", type=str)).upper()
                if val == "NA":
                    ind = 0
                else:
                    ind = self.ui.quad_combobox.findText(val)
                root.debug("Quad {1} index: {0}".format(ind, val))
                self.ui.quad_combobox.setCurrentIndex(ind)

                val = str(self.settings.value("section_screen", "NA", type=str)).upper()
                if val == "NA":
                    ind = 0
                else:
                    ind = self.ui.screen_combobox.findText(val)
                root.debug("Screen {1} index: {0}".format(ind, val))
                self.ui.screen_combobox.setCurrentIndex(ind)

            self.section_init_flag = False
            self.current_section = sect
        if len(quads) > 0:
            quad_name = str(self.ui.quad_combobox.currentText()).upper()
            # This will work since the combobox is populated in the same order as the stored section quadlist
            quad_sel = quads[self.ui.quad_combobox.currentIndex()]          # type: SectionQuad
            quad_length = quad_sel.length
            quad_pos = quad_sel.position
            self.ui.quad_length_label.setText("{0:.2f}".format(quad_length))
            self.ui.quad_combobox.blockSignals(False)
        else:
            quad_name = None
            quad_pos = 0
        if len(screens) > 0:
            screen_name = str(self.ui.screen_combobox.currentText()).upper()
            screen_sel = screens[self.ui.screen_combobox.currentIndex()]    # type: SectionScreen
            screen_pos = screen_sel.position
            # screen_pos = 0
            self.ui.screen_combobox.blockSignals(False)
        else:
            screen_name = None

        # Set the quad and screen selected:
        if quad_name is not None and screen_name is not None:
            if self.current_screen is None:
                self.set_section(quads[0], screens[0])
            else:
                if self.current_screen.name != screen_name or sect != self.current_section:
                    self.screen_init_flag = True
                # if self.current_quad.name != quad_name or self.current_screen.name != screen_name:
                if self.current_quad.name != quad_name or self.current_screen.name != screen_name \
                        or sect != self.current_section:
                    root.debug("New device selected.")
                    self.set_section(quad_sel, screen_sel)
            self.ui.quad_screen_dist_label.setText("{0:2f}".format(screen_pos - quad_pos))
            # self.ui.quad_screen_dist_label.setText("Quad pos {0}, Screen pos {1}".format(quad_pos, screen_pos))

    def set_section(self, new_quad, new_screen):
        """
        Setup hardware access to section from current_sect, current_quad, current_screen:

        Will add devices to the device handler for quad mag, crq + scrn, liveviewer, beamviewer

        Stop current monitor of k-value, image

        Start new monitor task of k-value, image

        :param new_quad:
        :param new_screen:
        :return:
        """
        root.info("Set section {0} with {1} and {2}".format(self.current_section,
                                                            new_quad.name,
                                                            new_screen.name))
        try:
            load_quad = new_quad.name != self.current_quad.name
        except AttributeError:
            load_quad = True
        if load_quad:
            for t in self.quad_tasks:
                t.cancel()
            self.quad_tasks = list()
            k_task = TangoReadAttributeTask("mainfieldcomponent", new_quad.crq, self.device_handler,
                                            name="k_read", callback_list=[self.read_k])
            # k_task.start()
            k_rep_task = RepeatTask(k_task, -1, 0.3, name="k_repeat")
            k_rep_task.start()
            self.quad_init_flag = True
            self.quad_tasks.append(k_rep_task)
            self.current_quad = new_quad
            self.ui.current_quad_sel_label.setText("{0}".format(new_quad.mag.upper()))
            # Add more device connections here
            e_task = TangoReadAttributeTask("energy", new_quad.crq, self.device_handler,
                                            name="e_read", callback_list=[self.read_k])
            e_task.start()

        try:
            load_screen = new_screen.name != self.current_screen.name
        except AttributeError:
            load_screen = True
        if load_screen:
            for t in self.screen_tasks:
                t.cancel()
            self.screen_tasks = list()
            image_task = TangoReadAttributeTask("image", new_screen.liveviewer, self.device_handler,
                                                name="cam_image_read", callback_list=[self.read_image])
            rep_task = RepeatTask(image_task, -1, 0.3, name="cam_image_repeat")
            rep_task.start()
            self.screen_tasks.append(rep_task)
            cam_state_task = TangoReadAttributeTask("state", new_screen.liveviewer, self.device_handler,
                                                    name="cam_state_read", callback_list=[self.read_image])
            rep_task = RepeatTask(cam_state_task, -1, 0.5, name="cam_state_repeat")
            rep_task.start()
            self.screen_tasks.append(rep_task)
            cam_state_task = TangoReadAttributeTask("framerate", new_screen.liveviewer, self.device_handler,
                                                    name="cam_reprate_read", callback_list=[self.read_image])
            rep_task = RepeatTask(cam_state_task, -1, 0.5, name="cam_reprate_repeat")
            rep_task.start()
            self.screen_tasks.append(rep_task)
            screen_in_task = TangoReadAttributeTask("statusin", new_screen.screen, self.device_handler,
                                                    name="screen_in_read", callback_list=[self.read_image])
            rep_task = RepeatTask(screen_in_task, -1, 0.5, name="screen_in_repeat")
            rep_task.start()
            self.screen_tasks.append(rep_task)
            cam_cal_task = BagOfTasksTask([TangoReadAttributeTask("measurementruler", new_screen.beamviewer,
                                                                  self.device_handler, name="cam_cal_ruler"),
                                           TangoReadAttributeTask("measurementrulerwidth", new_screen.beamviewer,
                                                                  self.device_handler, name="cam_cal_width"),
                                           TangoReadAttributeTask("roi", new_screen.beamviewer,
                                                                  self.device_handler, name="cam_cal_roi")
                                           ],
                                          name="cam_cal_read", callback_list=[self.read_image])
            cam_cal_task.start()
            self.current_screen = new_screen

            # Add more device connections here

    def populate_sections(self, task):
        """
        Callback from populate devices task, which goes through the tango database and checks for devices
        applicable to quadscans.

        :param task: Task object that sent the callback
        :return:
        """
        root.info("Populate section finished.")
        self.section_devices = task.get_result(wait=False)
        self.update_section()

    def update_scan_devices(self):
        root.info("Updating scan devices")

    def update_process_image_roi(self):
        """
        Callback for updating the ROI selection in the raw process image. When changed the roi spinboxes
        are updated and a process all images task is started.

        :return:
        """
        root.info("Updating ROI for process image")
        pos = self.ui.process_image_widget.roi.pos()
        size = self.ui.process_image_widget.roi.size()
        center = [pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0]
        self.ui.p_roi_cent_x_spinbox.blockSignals(True)
        self.ui.p_roi_cent_x_spinbox.setValue(center[0])
        self.ui.p_roi_cent_x_spinbox.blockSignals(False)
        self.ui.p_roi_cent_y_spinbox.blockSignals(True)
        self.ui.p_roi_cent_y_spinbox.setValue(center[1])
        self.ui.p_roi_cent_y_spinbox.blockSignals(False)
        self.ui.p_roi_size_w_spinbox.blockSignals(True)
        self.ui.p_roi_size_w_spinbox.setValue(size[0])
        self.ui.p_roi_size_w_spinbox.blockSignals(False)
        self.ui.p_roi_size_h_spinbox.blockSignals(True)
        self.ui.p_roi_size_h_spinbox.setValue(size[1])
        self.ui.p_roi_size_h_spinbox.blockSignals(False)
        # self.process_image_raw_roi = [pos[0], pos[1], size[0], size[1]]
        self.start_processing()

    def change_raw_or_processed_view(self):
        """
        Select which image to be shown in the process image widget.
        Raw is before thresholding, cropping, and median filtering.
        Filtered is after these operations.

        :return:
        """
        # Save current view:
        view_range = self.ui.process_image_widget.view.viewRange()
        pos = [view_range[0][0], view_range[1][0]]
        size = [view_range[0][1] - view_range[0][0], view_range[1][1] - view_range[1][0]]

        # Restore previous view:
        x_range = [self.process_image_view[0],
                   self.process_image_view[0] + self.process_image_view[2]]
        y_range = [self.process_image_view[1],
                   self.process_image_view[1] + self.process_image_view[3]]
        root.debug("x range: {0}, y range: {1}".format(x_range, y_range))
        self.ui.process_image_widget.view.setRange(xRange=x_range, yRange=y_range)

        self.process_image_view = [pos[0], pos[1], size[0], size[1]]

        self.update_image_selection()

    def change_analysis_axis(self):
        self.start_processing()

    def update_process_image_threshold(self):
        root.info("Updating image threshold from histogram widget")
        hl = self.ui.process_image_widget.getHistogramWidget().getLevels()
        root.debug("Levels: {0}".format(hl))
        self.ui.p_threshold_spinbox.blockSignals(True)
        self.ui.p_threshold_spinbox.setValue(hl[0])
        self.ui.p_threshold_spinbox.blockSignals(False)
        self.start_processing()

    def update_process_image_histogram(self):
        levels = self.ui.process_image_widget.getHistogramWidget().getLevels()
        root.info("Histogram changed: {0}".format(levels))
        self.ui.p_threshold_spinbox.setValue(levels[0])
        self.start_processing()

    def update_camera_roi(self):
        root.info("Updating ROI for camera image")

    def update_image_processing(self, task=None):
        if task is not None:
            if not task.is_done():
                proc_image_list = task.get_result(wait=False)
                root.debug("New image list: {0}".format(len(proc_image_list)))
                # root.debug("Sigma x: {0}".format([x.sigma_x for x in proc_image_list]))
                # root.debug("Im 0 thr: {0}".format(proc_image_list[0].threshold))
                if len(proc_image_list) > 0:
                    self.quad_scan_data_analysis = self.quad_scan_data_analysis._replace(proc_images=proc_image_list)
                    self.update_image_selection(None)
                self.start_fit()

    def update_image_selection(self, image=None, auto_levels=False, auto_range=False):
        if image is None or isinstance(image, int):
            im_ind = self.ui.p_image_index_slider.value()
            if self.ui.p_raw_image_radio.isChecked():
                # Raw image selected
                try:
                    image_struct = self.quad_scan_data_analysis.images[im_ind]
                except IndexError:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    msg = "Index {0} out of range, len {1}.".format(im_ind,
                                                                    len(self.quad_scan_data_analysis.images))
                    self.ui.status_textedit.append("\n---------------------------\n"
                                                   "{0}:\n"
                                                   "{1}\n".format(time_str, msg))
                    root.error(msg)

                    return
                image = image_struct.image
                try:

                    self.ui.process_image_widget.setImage(np.transpose(image), autoRange=auto_range, autoLevels=auto_levels)
                    self.ui.process_image_widget.roi.show()
                    self.ui.process_image_widget.update()
                except TypeError as e:
                    root.error("Error setting image: {0}".format(e))

            else:
                # Filtered image selected
                try:
                    image_struct = self.quad_scan_data_analysis.proc_images[im_ind]    # type: ProcessedImage
                except IndexError:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    msg = "Index {0} out of range, len {1}.".format(im_ind,
                                                                    len(self.quad_scan_data_analysis.proc_images))
                    self.ui.status_textedit.append("\n---------------------------\n"
                                                   "{0}:\n"
                                                   "{1}\n".format(time_str, msg))
                    root.error(msg)
                    return
                image = image_struct.pic_roi
                try:
                    self.ui.process_image_widget.roi.hide()
                    self.ui.process_image_widget.setImage(np.transpose(image), autoRange=False, autoLevels=auto_levels)
                except TypeError as e:
                    root.error("Error setting image: {0}".format(e))

            self.ui.p_k_value_label.setText(u"k = {0:.3f} 1/m\u00B2".format(image_struct.k_value))
            self.ui.p_k_ind_label.setText("k index {0}/{1}".format(image_struct.k_ind,
                                                                   self.quad_scan_data_analysis.acc_params.num_k - 1))
            self.ui.p_image_label.setText("image {0}/{1}".format(image_struct.image_ind,
                                                                 self.quad_scan_data_analysis.acc_params.num_images - 1))
        else:
            # If an image was sent directly to the method, such as when updating a loading task
            try:
                # self.ui.process_image_widget.setImage(image)
                self.ui.process_image_widget.setImage(np.transpose(image), autoRange=auto_range, autoLevels=auto_levels)
            except TypeError as e:
                root.error("Error setting image: {0}".format(e))

    def update_fit_result(self, task=None):
        root.info("{0}: Updating fit result.".format(self))
        if task is not None:
            if self.ui.p_x_radio.isChecked():
                self.ui.result_axis_label.setText("x-axis")
            else:
                self.ui.result_axis_label.setText("y-axis")

            fitresult = task.get_result(wait=False)     # type: FitResult
            if isinstance(fitresult, Exception):
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                self.ui.status_textedit.append("\n===================\n"
                                               "{0}:\n"
                                               "Could not generate fit: \n"
                                               "{1}\n".format(time_str, fitresult))
                self.ui.eps_label.setText("-- mm x mmrad")
                self.ui.beta_label.setText("-- m")
                self.ui.alpha_label.setText("--")
                self.fit_result = fitresult
                self.update_fit_signal.emit()

            else:
                if fitresult is not None:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    self.ui.status_textedit.append("\n===================\n"
                                                   "{0}:\n"
                                                   "Fit ok. \n"
                                                   "Residual: {1}\n".format(time_str, fitresult.residual))
                    self.ui.eps_label.setText("{0:.2f} mm x mmrad".format(1e6 * fitresult.eps_n))
                    self.ui.beta_label.setText("{0:.2f} m".format(fitresult.beta))
                    self.ui.alpha_label.setText("{0:.2f}".format(fitresult.alpha))
                    self.fit_result = fitresult
                    self.update_fit_signal.emit()
                else:
                    root.error("Fit result NONE")

    def update_camera_image(self, new_image):
        # root.debug("Updating camera image")
        if self.screen_init_flag:
            self.ui.camera_widget.setImage(np.transpose(new_image), autoRange=True, autoLevels=True)
            self.screen_init_flag = False
        else:
            self.ui.camera_widget.setImage(np.transpose(new_image), autoRange=False, autoLevels=False)
        self.ui.camera_widget.roi.show()
        self.ui.camera_widget.update()

    def plot_sigma_data(self):
        root.info("Plotting sigma data")
        use_x_axis = self.ui.p_x_radio.isChecked()
        if use_x_axis is True:
            sigma = np.array([proc_im.sigma_x for proc_im in self.quad_scan_data_analysis.proc_images])
        else:
            sigma = np.array([proc_im.sigma_y for proc_im in self.quad_scan_data_analysis.proc_images])
        k = np.array([proc_im.k_value for proc_im in self.quad_scan_data_analysis.proc_images])
        q = np.array([proc_im.q for proc_im in self.quad_scan_data_analysis.proc_images])
        en_data = np.array([proc_im.enabled for proc_im in self.quad_scan_data_analysis.proc_images])
        sigma_symbol_list = list()
        sigma_brush_list = list()
        so_brush = pq.mkBrush(150, 150, 250, 150)
        sx_brush = pq.mkBrush(250, 150, 150, 150)
        q_symbol_list = list()
        q_brush_list = list()
        qo_brush = pq.mkBrush(150, 250, 150, 150)
        qx_brush = pq.mkBrush(250, 100, 100, 200)
        for en in en_data:
            if not en:
                sigma_symbol_list.append("t")
                sigma_brush_list.append(sx_brush)
                q_symbol_list.append("t")
                q_brush_list.append(qx_brush)
            else:
                sigma_symbol_list.append("o")
                sigma_brush_list.append(so_brush)
                q_symbol_list.append("s")
                q_brush_list.append(qo_brush)

        self.sigma_x_plot.setData(x=k, y=sigma, symbol=sigma_symbol_list, brush=sigma_brush_list, size=10, pen=None)
        self.charge_plot.setData(x=k, y=q, symbol=q_symbol_list, brush=q_brush_list, size=10, pen=None)
        # self.charge_plot.setData(x=k, y=q, symbol=q_symbol_list, symbolBrush=q_brush_list,
        #                          symbolPen=None, pen=None)
        # y_range = [0, q.max()]
        # x_range = [k.min(), k.max()]
        # self.ui.charge_widget.getViewBox().setRange(xRange=x_range, yRange=y_range, disableAutoRange=True)

        fit_data = self.fit_result.fit_data
        if fit_data is not None:
            self.fit_x_plot.setData(x=fit_data[0], y=fit_data[1])
            self.ui.fit_widget.update()
            self.ui.charge_widget.update()

    def set_algo(self):
        root.info("Setting fit algo")
        self.start_fit()

    def set_start_k(self):
        root.info("Setting start k value to {0}".format(self.ui.k_current_spinbox.value()))
        self.ui.k_start_spinbox.setValue(self.ui.k_current_spinbox.value())

    def set_end_k(self):
        root.info("Setting end k value to {0}".format(self.ui.k_current_spinbox.value()))
        self.ui.k_end_spinbox.setValue(self.ui.k_current_spinbox.value())

    def set_current_k(self):
        value = self.ui.k_current_spinbox.value()
        root.info("Setting current k to {0}".format(value))
        self.ui.current_k_label.setText("k = {0:.3f} 1/m²".format(value))
        task = TangoWriteAttributeTask("mainfieldcomponent", self.current_quad.crq, self.device_handler, value,
                                       "write_k")
        task.start()

    def set_base_dir(self):
        root.info("Setting base save directory")

    def start_camera(self):
        root.info("Starting camera {0}".format(self.current_screen.liveviewer))
        task = TangoCommandTask("start", self.current_screen.liveviewer, self.device_handler)
        task.start()

    def stop_camera(self):
        root.info("Stopping camera {0}".format(self.current_screen.liveviewer))
        task = TangoCommandTask("stop", self.current_screen.liveviewer, self.device_handler)
        task.start()

    def insert_screen(self):
        root.info("Inserting screen {0}".format(self.current_screen.screen))
        task = TangoCommandTask("movein", self.current_screen.screen, self.device_handler)
        task.start()

    def remove_screen(self):
        root.info("Removing screen {0}".format(self.current_screen.screen))
        task = TangoCommandTask("moveout", self.current_screen.screen, self.device_handler)
        task.start()

    def start_scan(self):
        """
        Start a new scan
        :return:
        """
        root.info("Start scan pressed")
        if self.assert_scan_start_conditions():
            if self.generate_daq_info():
                self.scan_image_max = 0.0
                k0 = self.ui.k_start_spinbox.value()
                k1 = self.ui.k_end_spinbox.value()
                dk = (k1 - k0) / np.maximum(1, self.ui.num_k_spinbox.value() - 1)
                scan_param = ScanParam(scan_attr_name="mainfieldcomponent", scan_device_name=self.current_quad.crq,
                                       scan_start_pos=k0, scan_end_pos=k1, scan_step=dk,
                                       scan_pos_tol=np.maximum(dk*0.01, 0.001), scan_pos_check_interval=0.1,
                                       measure_attr_name_list=["image"], measure_device_list=[self.current_screen.liveviewer],
                                       measure_number=self.ui.num_images_spinbox.value(),
                                       measure_interval=self.ui.reprate_spinbox.value())
                # callback_list=[self.scan_callback] is called for each completed step.
                # read_callback=self.scan_image_callback is called for every measurement (image taken)
                self.scan_task = TangoScanTask(scan_param=scan_param, device_handler=self.device_handler, name="scan",
                                               timeout=5.0, callback_list=[self.scan_callback],
                                               read_callback=self.scan_image_callback)
                self.scan_task.start()
                if self.ui.update_analysis_radiobutton.isChecked():
                    # Should also update roi from camera roi
                    source_name = "Scan data {0}-{1}".format(self.current_quad.mag, self.current_screen.screen)
                    self.ui.data_source_label.setText(source_name)

                    self.quad_scan_data_analysis = self.quad_scan_data_scan
                    self.update_analysis_parameters()
                    self.image_processor.clear_callback_list()
                    self.image_processor.add_callback(self.scan_image_processed_callback)
                root.info("Scan started. Parameters: {0}".format(scan_param))
            else:
                root.error("Scan not started. Could not generate daq_info")
        else:
            root.error("Scan not started. Start conditions not met.")

    def stop_scan(self):
        root.info("Stop scan pressed")
        self.ui.scan_status_label.setText("STOPPED: k -/- image -/-")
        self.ui.scan_progress_label.setText("[----------]")
        if self.scan_task is not None:
            self.scan_task.cancel()

    def assert_scan_start_conditions(self):
        """
        Check pre-conditions necessary to start a scan. Return True if ok.
        Will also cancel an existing running scan task.

        - Is the camera running
        - Is the screen in

        TODO:
        - Is the quad on
        - Are the quads between screen and quad off
        - Is the screen after the quad

        :return: True if scan can be started, False otherwise.
        """
        root.info("Checking start conditions (camera running, screen in, quad on")
        if self.scan_task is not None:
            self.scan_task.cancel()
        if str(self.ui.camera_state_label.text()).upper() not in ["RUNNING", "ON"]:
            root.warning("Camera not running. Can't start scan")
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n------------------------\n"
                                           "{0}:\n"
                                           "Camera not running. Can't start scan.\n".format(time_str))
            return False
        if "IN" not in str(self.ui.screen_state_label.text()).upper():
            root.warning("Screen not inserted. Can't start scan")
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n------------------------\n"
                                           "{0}:\n"
                                           "Screen not inserted. Can't start scan.\n".format(time_str))
            return False
        return True

    def generate_daq_info(self):
        """
        Generate daq_info.txt file and AcceleratorParameters.
        Init quad_scan_data_scan with these parameters.
        :return: True is success
        """
        root.info("Generating daq_info")
        k0 = self.ui.k_start_spinbox.value()
        k1 = self.ui.k_end_spinbox.value()
        dk = (k1 - k0) / np.maximum(1, self.ui.num_k_spinbox.value() - 1)

        roi_size = self.ui.camera_widget.roi.size()
        roi_pos = self.ui.camera_widget.roi.pos()
        roi_center = [roi_pos[0] + roi_size[0] / 2.0, roi_pos[1] + roi_size[1] / 2.0]
        roi_dim = [roi_size[0], roi_size[1]]
        root.info("ROI: pos {0}, size {1}, center {2}".format(roi_pos, roi_size, roi_center))
        acc_params = AcceleratorParameters(electron_energy=self.ui.electron_energy_spinbox.value(),
                                           quad_length=float(self.ui.quad_length_label.text()),
                                           quad_screen_dist=float(self.ui.quad_screen_dist_label.text()),
                                           k_max=k1, k_min=k0, cal=self.camera_cal,
                                           num_k=self.ui.num_k_spinbox.value(),
                                           num_images=self.ui.num_images_spinbox.value(),
                                           quad_name=self.current_quad.mag,
                                           screen_name=self.current_screen.screen,
                                           roi_center=roi_center,
                                           roi_dim=roi_dim)
        self.quad_scan_data_scan = QuadScanData(acc_params=acc_params, images=[], proc_images=[])

        # Save daq_info.txt:

        save_dict = OrderedDict()
        try:
            save_dict["main_dir"] = self.ui.save_path_linedit.text()
            save_dict["daq_dir"] = self.ui.save_path_linedit.text()
            save_dict["quad"] = self.current_quad.mag
            save_dict["quad_length"] = "{0}".format(self.ui.quad_length_label.text())
            save_dict["quad_2_screen"] = "{0}".format(self.ui.quad_screen_dist_label.text())
            save_dict["screen"] = self.current_screen.screen
            save_dict["pixel_dim"] = "{0} {1}".format(self.camera_cal[0], self.camera_cal[1])
            save_dict["num_k_values"] = "{0}".format(self.ui.num_k_spinbox.value())
            save_dict["num_shots"] = "{0}".format(self.ui.num_images_spinbox.value())
            save_dict["k_min"] = "{0}".format(k0)
            save_dict["k_max"] = "{0}".format(k1)
            val = roi_center
            save_dict["roi_center"] = "{0} {1}".format(val[0], val[1])
            val = roi_dim
            save_dict["roi_dim"] = "{0} {1}".format(val[0], val[1])
            save_dict["beam_energy"] = "{0}".format(self.ui.electron_energy_spinbox.value())
            save_dict["camera_bpp"] = 16
        except KeyError as e:
            msg = "Could not generate daq_info: {0}".format(e)
            root.exception(e)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n=============================\n"
                                           "{0}:\n"
                                           "Could not generate daq_info: {1}\n".format(time_str, e))
            return False
        except IndexError as e:
            msg = "Could not generate daq_info: {0}".format(e)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n=============================\n"
                                           "{0}:\n"
                                           "Could not generate daq_info: {1}\n".format(time_str, e))
            root.exception(e)
            return False
        base_path = str(self.ui.save_path_linedit.text())
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        dev_str = "{0}_{1}".format(self.current_quad.mag, self.current_screen.screen)
        save_path = "{0}/{1}_{2}".format(base_path, time_str, dev_str.replace("/", "-"))
        try:
            os.makedirs(save_path)
            self.scan_save_path = save_path
        except OSError as e:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n=============================\n"
                                           "{0}:\n"
                                           "Error creating directory: {1}\n".format(time_str, e))
            return False
        full_name = os.path.join(save_path, "daq_info.txt")
        with open(full_name, "w+") as f:
            for key, value in save_dict.iteritems():
                s = "{0} : {1}\n".format(key.ljust(13, " "), value)
                f.write(s)
            f.write("***** Starting loop over quadrupole k-values *****\n")
            f.write("+------+-------+----------+----------+----------------------+\n")
            f.write("|  k   |  shot |    set   |   read   |        saved         |\n")
            f.write("|  #   |   #   |  k-value |  k-value |     image file       |\n")
        return True

    def scan_callback(self, task):
        """
        Callback for each step of the scan and also when the scan is completed.
        Update scan info.

        :param task: scan task
        :return:
        """
        root.debug("Scan callback")
        if not task.is_done():
            res = task.get_result(wait=False)       # Result contains: [write_pos_res, read_pos_res, measure_list]
            if not task.is_cancelled():
                try:
                    pos = res[1].value
                    timestamp = res[1].time
                except TypeError as e:
                    root.exception("Scan callback result error: ")
                # measure_list = res[2]
                # quad_image_list = [QuadImage(k_ind=0, k_value=pos, image_ind=ind, image=im)
                #                    for ind, im in enumerate(measure_list)]
                # images = self.quad_scan_data_scan.images + quad_image_list
                # self.quad_scan_data_scan._replace(images=images)
        else:
            self.ui.scan_status_label.setText("DONE")
        # if self.ui.update_analysis_radiobutton.isChecked():
            # self.quad_scan_data_analysis = self.quad_scan_data_scan
            # self.update_analysis_parameters()
            # self.update_image_selection()
            # self.update_fit_signal.emit()
            # self.start_fit()

    def scan_image_callback(self, task):
        """
        Callback for a new image that had been read. This image is saved and shown in the GUI.
        The name of the task contains the index of k and image, and k value.

        :param task: TangoReadAttributeTask returning an image.
        :return:
        """
        root.debug("Scan image callback")
        name_elements = task.get_name().split("_")
        try:
            image = task.get_result(wait=False).value
        except AttributeError as e:
            root.error("Scan image not valid task. {0}".format(e))
            return

        try:
            im_ind = int(name_elements[4])
            num_images = self.quad_scan_data_scan.acc_params.num_images
            k_ind = int(name_elements[2])
            num_k = self.quad_scan_data_scan.acc_params.num_k
            k_value = float(name_elements[3])
            root.info("Scan image {0} {1} (size {2}), sending for processing".format(k_ind, im_ind, image.shape))
            self.update_camera_signal.emit(image)
            # self.ui.camera_widget.setImage(image, autoLevels=False, autoRange=False)
            quadimage = QuadImage(k_ind=k_ind, k_value=k_value, image_ind=im_ind, image=image)
            # Appending image to images list in "immutable" named tuple....... :)
            self.quad_scan_data_scan.images.append(quadimage)
            task = SaveQuadImageTask(quadimage, save_path=str(self.scan_save_path),
                                     name="scan_save_{0}".format(str(self.ui.save_name_lineedit.text())),
                                     callback_list=[self.save_image_callback])
            task.start()

            s = "RUNNING: k {0}/{1} image {2}/{3}".format(k_ind+1, float(num_k), im_ind+1, float(num_images))
            self.ui.scan_status_label.setText(s)
            p = int((k_ind + (im_ind + 1) / float(num_images)) / float(num_k) * 10)
            # root.debug("p={0}".format(p))
            self.ui.scan_progress_label.setText("[{0}{1}]".format("="*p, "-"*(10-p)))

            try:
                self.scan_image_max = np.maximum(self.scan_image_max, np.max(image))
            except Exception as e:
                root.exception("Error when np.maximum")

            # Put image for processing is update while scan is selected:
            if self.ui.update_analysis_radiobutton.isChecked():
                self.image_processor.set_roi(self.quad_scan_data_scan.acc_params.roi_center,
                                             self.quad_scan_data_scan.acc_params.roi_dim)
                threshold = self.ui.p_threshold_spinbox.value()
                kernel = self.ui.p_median_kernel_spinbox.value()
                self.image_processor.set_processing_parameters(threshold, self.quad_scan_data_scan.acc_params.cal, kernel)
                root.debug("Starting processing quadimage {0}".format(quadimage))
                self.image_processor.process_image(quadimage, enabled=True)

                # Show image if raw radio is selected:
                ind = k_ind + self.quad_scan_data_scan.acc_params.num_k * im_ind
                self.ui.p_image_index_slider.blockSignals(True)
                self.ui.p_image_index_slider.setMaximum(ind)
                self.ui.p_image_index_slider.setValue(ind)
                self.ui.p_image_index_slider.blockSignals(False)
                if self.ui.p_raw_image_radio.isChecked():

                    self.update_image_selection(auto_levels=True)

        except IndexError as e:
            root.exception("Error for returned image in scan")
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n{0}: \n"
                                           "Error for returned image in scan\n".format(time_str))
            return
        except ValueError as e:
            root.exception("Error for returned image name in scan")
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n{0}: \n"
                                           "Error for returned image name in scan\n{1}\n".format(time_str,
                                                                                                        name_elements))
            return

        self.quad_scan_data_analysis = self.quad_scan_data_scan

    def scan_image_processed_callback(self, task):
        root.debug("Scan image processed.")
        proc_image = task.get_result(wait=False)    # type: ProcessedImage
        ind = proc_image.k_ind + self.quad_scan_data_scan.acc_params.num_k * proc_image.image_ind
        self.quad_scan_data_scan.proc_images.append(proc_image)
        self.quad_scan_data_analysis = self.quad_scan_data_scan

        if not self.ui.p_raw_image_radio.isChecked():
            root.debug("Image {0} {1}, index {2}".format(proc_image.k_ind, proc_image.image_ind, ind))
            self.ui.p_image_index_slider.blockSignals(True)
            self.ui.p_image_index_slider.setValue(ind)
            self.ui.p_image_index_slider.blockSignals(False)
            self.update_image_selection(auto_levels=True)

        # self.ui.process_image_widget.setImage(proc_image.pic_roi)
        self.start_fit()

    def save_image_callback(self, task):
        result = task.get_result(wait=False)
        root.debug("Save image task {0} returned {1}".format(task.get_name(), result))
        if task.is_cancelled():
            root.exception("Error for save image in scan")
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            self.ui.status_textedit.append("\n{0}: \n"
                                           "Error for save image in scan\n{1}\n".format(time_str,
                                                                                               result))
            return

    def start_processing(self):
        # if len(self.processing_tasks) > 0:
        #     for task in self.processing_tasks:
        #         root.info("Removing processing task {0}".format(task.get_name()))
        #         task.cancel()
        #         self.processing_tasks.remove(task)
        th = self.ui.p_threshold_spinbox.value()
        kern = self.ui.p_median_kernel_spinbox.value()
        keep_charge_ratio = 0.01 * self.ui.p_keep_charge_ratio_spinbox.value()
        root.info("Start processing. Threshold: {0}, Kernel: {1}".format(th, kern))
        acc_params = self.quad_scan_data_analysis.acc_params         # type: AcceleratorParameters
        roi_center = [self.ui.p_roi_cent_y_spinbox.value(), self.ui.p_roi_cent_x_spinbox.value()]
        roi_size = [self.ui.p_roi_size_h_spinbox.value(), self.ui.p_roi_size_w_spinbox.value()]
        if acc_params is not None:
            acc_params = acc_params._replace(roi_center=roi_center)
            acc_params = acc_params._replace(roi_dim=roi_size)
            self.quad_scan_data_analysis = self.quad_scan_data_analysis._replace(acc_params=acc_params)
            self.quad_scan_data_analysis = self.quad_scan_data_analysis._replace(acc_params=acc_params)
        # root.info("Start processing. Accelerator params: {0}".format(acc_params))
        try:
            root.info("Start processing. Num images: {0}".format(len(self.quad_scan_data_analysis.images)))
        except TypeError:
            root.info("No images.")
            return
        # task = ProcessAllImagesTask(self.quad_scan_data_analysis, threshold=th,
        #                             kernel_size=kern,
        #                             image_processor_task=self.image_processor,
        #                             process_exec_type="process",
        #                             name="process_images",
        #                             callback_list=[self.update_image_processing])
        self.image_processor.clear_callback_list()
        self.image_processor.add_callback(self.update_image_processing)
        self.image_processor.process_images(self.quad_scan_data_analysis,
                                            threshold=th, kernel=kern, enabled_list=self.user_enable_list,
                                            keep_charge_ratio=keep_charge_ratio)
        # task.start()
        # self.processing_tasks.append(task)

    def start_fit(self):
        if "Full matrix" in str(self.ui.fit_algo_combobox.currentText()):
            algo = "full"
        else:
            algo = "thin lens"
        if self.ui.p_x_radio.isChecked():
            axis = "x"
        else:
            axis = "y"
        task = FitQuadDataTask(self.quad_scan_data_analysis.proc_images,
                               self.quad_scan_data_analysis.acc_params,
                               algo=algo, axis=axis, name="fit")
        task.add_callback(self.update_fit_result)
        task.start()

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
            # Catch if the click did not hit a point:
            pos = point_list[0].pos()
            root.info("Point clicked: {0}".format(pos))
        except IndexError:
            root.debug("No points in list - exit")
            return
        root.debug("Right button: {0}".format(right))

        # Check which plot was clicked:
        if scatterplotitem == self.charge_plot:
            y = [proc_im.q for proc_im in self.quad_scan_data_analysis.proc_images]
        else:
            use_x_axis = self.ui.p_x_radio.isChecked()
            if use_x_axis is True:
                y = [proc_im.sigma_x for proc_im in self.quad_scan_data_analysis.proc_images]
            else:
                y = [proc_im.sigma_y for proc_im in self.quad_scan_data_analysis.proc_images]

        x = [proc_im.k_value for proc_im in self.quad_scan_data_analysis.proc_images]
        # en_data = [proc_im.enabled for proc_im in self.quad_scan_data_analysis.proc_images]
        en_data = self.user_enable_list
        enabled = not right             # True if left button is pressed
        eps = 1e-9
        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        root.debug("Mouse pos: {0}".format(mouse_pos))
        k_sel_i = None
        sel_dist = np.inf
        k_toggle_i = None
        tog_dist = np.inf
        for p in point_list:
            pos = p.pos()
            # We need to loop through the list of data points to find the index of the points clicked:
            for k_i, k_val in enumerate(x):
                    # Check if the point is within eps:
                    if abs(pos.x() - k_val) < eps:
                        if abs(pos.y() - (y[k_i])) < eps:
                            d = (pos.x() - mouse_pos.x())**2 + (pos.y() - mouse_pos.y())**2
                            if d < sel_dist:
                                k_sel_i = k_i
                                sel_dist = d
                            if en_data[k_i] != enabled:
                                if d < tog_dist:
                                    k_toggle_i = k_i
                                    tog_dist = d

        if k_toggle_i is not None:
            en_data[k_toggle_i] = enabled
            proc_im = self.quad_scan_data_analysis.proc_images[k_toggle_i]
            proc_im = proc_im._replace(enabled=enabled)
            proc_image_list = self.quad_scan_data_analysis.proc_images
            proc_image_list[k_toggle_i] = proc_im
            self.quad_scan_data_analysis = self.quad_scan_data_analysis._replace(proc_images=proc_image_list)

            self.ui.fit_widget.update()

        if k_sel_i is not None:
            self.ui.p_image_index_slider.setValue(k_sel_i)

            self.update_image_selection()
            self.start_fit()

    def enable_all_points(self):
        """
        Enable all points in processed data as a response from button press.

        :return:
        """
        root.info("Enable all points in processed data")
        self.user_enable_list = [True for x in range(len(self.user_enable_list))]
        self.start_processing()

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

    def read_k(self, task):
        try:
            name = task.get_name()
            result = task.get_result(wait=False)
        except AttributeError as e:
            root.warning("{0}: Not valid task... {1}".format(name, e))
            return

        if "k_read" in name:
            k = task.get_result(wait=False)
            self.ui.current_k_label.setText(u"k={0:.2f} 1/m\u00B2".format(k.value))
            if self.quad_init_flag:
                self.ui.k_current_spinbox.setValue(k.value)
                self.quad_init_flag = False
        elif "e_read" in name:
            e = task.get_result(wait=False)
            self.ui.electron_energy_spinbox.setValue(e.value * 1e-6)

    def read_image(self, task):
        """
        Callback for camera related tango attributes

        :param task: Task instance sending the callback
        :return:
        """
        # root.debug("Task {0} returning data".format(name))
        try:
            name = task.get_name()
            result = task.get_result(wait=False)
        except AttributeError as e:
            root.warning("{0}: Not valid task... {1}".format(name, e))
            return

        if "cam_image_read" in name:
            if task.is_cancelled():
                root.info("Image task cancelled.")

                for t in self.screen_tasks:
                    root.info("{0}: cancel state: {1}".format(t.get_name(), t.is_cancelled()))
                    if t.is_cancelled():
                        root.info("{0} cancelled.".format(t.name))
                        t.cancelled = False
                        t.start()
            else:
                self.update_camera_signal.emit(result.value)
        elif "cam_state_read" in name:
            self.ui.camera_state_label.setText("{0}".format(str(result.value)).upper())
        elif "cam_reprate_read" in name:
            self.ui.reprate_label.setText("{0:.1f} Hz".format(result.value))
        elif "screen_in_read" in name:
            if result.value:
                self.ui.screen_state_label.setText("{0} IN".format(str(self.ui.screen_combobox.currentText())))
            else:
                self.ui.screen_state_label.setText("{0} OUT".format(str(self.ui.screen_combobox.currentText())))
        elif "cam_cal_read" in name:
            try:
                meas_rul = eval(result[0].value)
                meas_w = result[1].value
                cal = meas_w / meas_rul["size"][0]
                roi = eval(result[2].value)
                root.debug("Camera ROI: {0}".format(roi))
                self.ui.camera_widget.roi.setPos([roi[0], roi[2]])
                self.ui.camera_widget.roi.setSize([roi[1]-roi[0], roi[3]-roi[2]])
            except TypeError as e:
                s = "Could not read calibration. Got {0}".format(result)
                root.exception(s)
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                self.ui.status_textedit.append("\n{0}: \n"
                                               "{1}\n".format(time_str, s))
                return
            root.debug("Camera calibration: {0} mm/pixel".format(cal))
            self.camera_cal = [cal, cal]
        else:
            root.error("Task {0} not useful for camera updating".format(name))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    root.info("QuadScanGui object created")
    myapp.show()
    root.info("App show")
    sys.exit(app.exec_())
    root.info("App exit")

