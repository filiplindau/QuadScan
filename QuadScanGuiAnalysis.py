# -*- coding: utf-8 -*-
"""
Created 2019-08-19

Gui with splitters to set relative size of areas.

@author: Filip Lindau
"""

from PyQt4 import QtGui, QtCore

import pyqtgraph as pq
import sys
import glob
import copy
import numpy as np
import itertools
from quadscan_gui_analysis import Ui_QuadScanDialog
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
root.setLevel(logging.INFO)

pq.graphicsItems.GradientEditorItem.Gradients['greyclip2'] = {
    'ticks': [(0.0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}
pq.graphicsItems.GradientEditorItem.Gradients['thermalclip'] = {
    'ticks': [(0, (0, 0, 75, 255)), (0.0001, (0, 0, 0, 255)), (0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
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
    update_proc_image_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        root.debug("Init")
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'QuadScanAnalysis')

        self.current_state = "unknown"
        self.last_load_dir = "."
        self.data_base_dir = "."
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

        self.process_image_proxy = None  # Signal proxy to track mouse position over image
        self.scan_proc_proxy = None  # Signal proxy to track mouse position over image

        self.quad_scan_data_analysis = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.quad_scan_data_scan = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.fit_result = FitResult(poly=None, alpha=None, beta=None, eps=None, eps_n=None,
                                    gamma_e=None, fit_data=None, residual=None)
        self.processing_tasks = list()

        self.gui_lock = threading.Lock()
        self.image_lock = threading.Lock()

        self.image_processor = None         # type: ProcessAllImagesTask2

        self.ui = Ui_QuadScanDialog()
        self.ui.setupUi(self)

        self.setup_layout()

        self.init_processing()

        root.info("Exit gui init")

    def init_processing(self):
        root.info("Initializing data structures")
        self.current_state = "unknown"
        # self.last_load_dir = "."
        # self.data_base_dir = "."
        self.load_init_flag = False     # Set when staring new load from disk

        self.load_image_max = 0.0
        self.scan_image_max = 0.0
        self.user_enable_list = list()

        self.quad_scan_data_analysis = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.quad_scan_data_scan = QuadScanData(acc_params=None, images=None, proc_images=None)
        self.fit_result = FitResult(poly=None, alpha=None, beta=None, eps=None, eps_n=None,
                                    gamma_e=None, fit_data=None, residual=None)
        self.processing_tasks = list()

        if self.image_processor is not None:
            self.image_processor.finish_processing()
        self.image_processor = ProcessAllImagesTask2(image_size=[2000, 2000], name="gui_image_proc",
                                                     callback_list=[self.update_image_processing])
        self.image_processor.start()

        self.ui.p_electron_energy_label.setText("--- MeV")
        self.ui.p_quad_length_label.setText("-.-- m")
        self.ui.p_quad_screen_dist_label.setText("-.-- m")
        self.ui.p_k_value_label.setText(u"k = -.-- 1/m\u00B2")
        self.ui.p_k_ind_label.setText("k index 0/0")
        self.ui.p_image_label.setText("image 0/0")
        self.ui.data_source_label.setText("--- No data loaded ---")
        self.ui.p_image_index_slider.setMaximum(0)
        self.ui.p_image_index_slider.setValue(0)
        self.ui.p_image_index_slider.update()
        self.update_image_selection(np.random.random((64, 64)), auto_levels=True, auto_range=True)
        self.update_fit_result()
        self.append_status_message("Init data structures")

    def setup_layout(self):
        """
        Setup GUI layout and set stored settings
        :return:
        """
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

        doc = self.ui.status_textedit.document()
        doc.setMaximumBlockCount(100)

        # This is to make sure . is the decimal character
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))

        # Restore settings
        self.last_load_dir = self.settings.value("load_path", ".", type=str)
        val = self.settings.value("threshold", "0.0", type=float)
        self.ui.p_threshold_spinbox.setValue(val)
        val = self.settings.value("keep_charge_ratio", "100.0", type=float)
        self.ui.p_keep_charge_ratio_spinbox.setValue(val)
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

        self.ui.process_image_widget.roi.sigRegionChangeFinished.connect(self.update_process_image_roi)
        hw = self.ui.process_image_widget.getHistogramWidget()
        # hw.sigLevelChangeFinished.connect(self.update_process_image_histogram)
        hw.item.blockSignals(True)
        self.ui.process_button.clicked.connect(self.start_processing)
        self.ui.init_button.clicked.connect(self.init_processing)
        self.ui.p_threshold_spinbox.editingFinished.connect(self.update_process_threshold_from_spinbox)
        self.ui.p_keep_charge_ratio_spinbox.editingFinished.connect(self.start_processing)
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
        self.ui.p_roi_cent_x_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_cent_y_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_w_spinbox.editingFinished.connect(self.set_roi)
        self.ui.p_roi_size_h_spinbox.editingFinished.connect(self.set_roi)

        self.update_fit_signal.connect(self.plot_sigma_data)
        self.update_proc_image_signal.connect(self.update_image_selection)

        # self.controller.image_done_signal.connect(self.update_fit_data)

        # Geometry setup
        window_pos_x = self.settings.value('window_pos_x', 100, type=int)
        window_pos_y = self.settings.value('window_pos_y', 100, type=int)
        window_size_w = self.settings.value('window_size_w', 1100, type=int)
        window_size_h = self.settings.value('window_size_h', 800, type=int)
        if window_pos_y < 50:
            window_pos_y = 50
        self.setGeometry(window_pos_x, window_pos_y, window_size_w, window_size_h)

        analysis_pic_plot_splitter_sizes = self.settings.value("analysis_pic_plot_splitter", [None], type="QVariantList")
        if analysis_pic_plot_splitter_sizes[0] is not None:
            self.ui.analysis_pic_plot_splitter.setSizes([np.int(s) for s in analysis_pic_plot_splitter_sizes])

        analysis_plots_splitter_sizes = self.settings.value("analysis_plots_splitter", [None], type="QVariantList")
        if analysis_plots_splitter_sizes[0] is not None:
            self.ui.analysis_plots_splitter.setSizes([np.int(s) for s in analysis_plots_splitter_sizes])

        # Setup signal proxies for mouse tracking
        self.process_image_proxy = pq.SignalProxy(self.ui.process_image_widget.scene.sigMouseMoved,
                                                  rateLimit=30, slot=self.process_image_mouse_moved)

    def eventFilter(self, obj, event):
        """
        Used for intercepting wheel events to modify magnet k-value
        :param obj:
        :param event:
        :return:
        """
        pass

    def closeEvent(self, event):
        """
        Closing the applications. Stopping threads and saving the settings.
        :param event:
        :return:
        """
        self.image_processor.clear_callback_list()
        root.info("Stop image processor")
        self.image_processor.finish_processing()
        root.info("Command sent.")
        self.settings.setValue("load_path", self.last_load_dir)

        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))

        self.settings.setValue("analysis_pic_plot_splitter", self.ui.analysis_pic_plot_splitter.sizes())
        self.settings.setValue("analysis_plots_splitter", self.ui.analysis_plots_splitter.sizes())

        self.settings.setValue("threshold", self.ui.p_threshold_spinbox.value())
        self.settings.setValue("keep_charge_ratio", self.ui.p_keep_charge_ratio_spinbox.value())
        self.settings.setValue("median_kernel", self.ui.p_median_kernel_spinbox.value())
        self.settings.setValue("filtered_image_show", self.ui.p_filtered_image_radio.isChecked())
        self.settings.setValue("use_x_axis", self.ui.p_x_radio.isChecked())

        if "Full matrix" in str(self.ui.fit_algo_combobox.currentText()):
            algo = "full matrix"
        else:
            algo = "thin lens"
        self.settings.setValue("fit_algo", algo)
        root.info("Settings done.")
        t0 = time.time()
        while not self.image_processor.is_done():
            time.sleep(0.1)
            if time.time() - t0 > 3:
                root.warning("Timeout waiting for image processes to finish")
                break
        event.accept()

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
        root.info("Loading data from {0}".format(self.last_load_dir))
        filedialog = OpenScanFileDialog(self.last_load_dir)
        g = self.geometry()
        filedialog.setGeometry(g.left()+20, g.top()+20, 1000, 700)
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

    def load_dir_entered(self, load_dir):
        """
        Set title of calling dialog window to show directory information
        in the form of number of images, quad, k values, electron energy.

        :param load_dir:
        :return:
        """
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
        root.debug("Update load data {0}, {1}".format(task.name, self.load_init_flag))
        if task is not None:
            result = task.get_result(wait=False)
            if isinstance(result, QuadImage):
            # if task.is_done() is False:
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
                        with self.image_lock:
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

    def update_process_threshold_from_spinbox(self):
        root.info("Updating image threshold from spinbox widget")
        hw = self.ui.process_image_widget.getHistogramWidget()
        hl = hw.getLevels()
        th = self.ui.p_threshold_spinbox.value()
        hw.blockSignals(True)
        hw.setLevels(th, hl[1])
        hw.blockSignals(False)
        self.start_processing()

    def update_process_image_histogram(self):
        levels = self.ui.process_image_widget.getHistogramWidget().getLevels()
        root.info("Histogram changed: {0}".format(levels))
        self.ui.p_threshold_spinbox.setValue(levels[0])
        self.start_processing()

    def update_image_processing(self, task=None):
        if task is not None:
            if not task.is_done():
                proc_image_list = task.get_result(wait=False)
                root.debug("New image list: {0}".format(len(proc_image_list)))
                # root.debug("Sigma x: {0}".format([x.sigma_x for x in proc_image_list]))
                # root.debug("Im 0 thr: {0}".format(proc_image_list[0].threshold))
                if len(proc_image_list) > 0:
                    with self.image_lock:
                        self.quad_scan_data_analysis = self.quad_scan_data_analysis._replace(proc_images=proc_image_list)
                    self.update_proc_image_signal.emit(None)
                    # self.update_image_selection(None)
                self.start_fit()

    def update_image_selection(self, image=None, auto_levels=False, auto_range=False):
        if image is None or isinstance(image, int):
            im_ind = self.ui.p_image_index_slider.value()
            if self.ui.p_raw_image_radio.isChecked():
                # Raw image selected
                try:
                    with self.image_lock:
                        image_struct = self.quad_scan_data_analysis.images[im_ind]
                        image = np.copy(image_struct.image)
                except IndexError:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    msg = "Index {0} out of range, len {1}.".format(im_ind,
                                                                    len(self.quad_scan_data_analysis.images))
                    self.ui.status_textedit.append("\n---------------------------\n"
                                                   "{0}:\n"
                                                   "{1}\n".format(time_str, msg))
                    root.error(msg)

                    return
                try:

                    self.ui.process_image_widget.setImage(np.transpose(image), autoRange=auto_range, autoLevels=auto_levels)
                    self.ui.process_image_widget.roi.show()
                    self.ui.process_image_widget.update()
                except TypeError as e:
                    root.error("Error setting image: {0}".format(e))

            else:
                # Filtered image selected
                try:
                    with self.image_lock:
                        image_struct = self.quad_scan_data_analysis.proc_images[im_ind]    # type: ProcessedImage
                        image = np.copy(image_struct.pic_roi)
                except IndexError:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    msg = "Index {0} out of range, len {1}.".format(im_ind,
                                                                    len(self.quad_scan_data_analysis.proc_images))
                    self.ui.status_textedit.append("\n---------------------------\n"
                                                   "{0}:\n"
                                                   "{1}\n".format(time_str, msg))
                    root.error(msg)
                    return

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
                self.ui.result_axis_label.setText("x-axis, "
                                                  "{0:.1f}% charge".format(self.ui.p_keep_charge_ratio_spinbox.value()))
            else:
                self.ui.result_axis_label.setText("y-axis, "
                                                  "{0:.1f}% charge".format(self.ui.p_keep_charge_ratio_spinbox.value()))

            fitresult = task.get_result(wait=False)     # type: FitResult
            if isinstance(fitresult, Exception):
                self.append_status_message("Could not generate fit.", fitresult)
                self.ui.eps_label.setText("-- mm x mmrad")
                self.ui.beta_label.setText("-- m")
                self.ui.alpha_label.setText("--")
                self.fit_result = fitresult
                self.update_fit_signal.emit()

            else:
                if fitresult is not None:
                    self.append_status_message("Fit ok.\n"
                                               "Residual: {0}".format(fitresult.residual))
                    self.ui.eps_label.setText("{0:.2f} mm x mmrad".format(1e6 * fitresult.eps_n))
                    self.ui.beta_label.setText("{0:.2f} m".format(fitresult.beta))
                    self.ui.alpha_label.setText("{0:.2f}".format(fitresult.alpha))
                    self.fit_result = fitresult
                    self.update_fit_signal.emit()
                else:
                    root.error("Fit result NONE")
        else:
            self.ui.eps_label.setText("-- mm x mmrad")
            self.ui.beta_label.setText("-- m")
            self.ui.alpha_label.setText("--")
            self.fit_result = None
            self.fit_x_plot.setData(x=[], y=[])
            self.sigma_x_plot.setData(x=[], y=[])
            self.charge_plot.setData(x=[], y=[])
            self.ui.fit_widget.update()
            self.ui.charge_widget.update()

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

        if not isinstance(self.fit_result, Exception):
            fit_data = self.fit_result.fit_data
            if fit_data is not None:
                self.fit_x_plot.setData(x=fit_data[0], y=fit_data[1])
                self.ui.fit_widget.update()
                self.ui.charge_widget.update()
        else:
            self.fit_x_plot.setData(x=[], y=[])
            self.ui.fit_widget.update()
            self.ui.charge_widget.update()

    def set_algo(self):
        root.info("Setting fit algo")
        self.start_fit()

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
            with self.image_lock:
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
            with self.image_lock:
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

    def append_status_message(self, msg, exception=None):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        if exception is None:
            self.ui.status_textedit.append("\n===================\n"
                                           "\n{0}: \n"
                                           "{1}\n".format(time_str, msg))
        else:
            self.ui.status_textedit.append("\n===================\n"
                                           "\n{0}: \n"
                                           "--ERROR--\n"
                                           "{1}\n"
                                           "{2}\n".format(time_str, msg, exception))
        self.ui.status_textedit.verticalScrollBar().setValue(self.ui.status_textedit.verticalScrollBar().maximum())

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
                "({0}, {1}) px: {2:.0f}".format(min(x, 9999), min(y, 9999), intensity))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = QuadScanGui()
    root.info("QuadScanGui object created")
    myapp.show()
    root.info("App show")
    sys.exit(app.exec_())
    root.info("App exit")

