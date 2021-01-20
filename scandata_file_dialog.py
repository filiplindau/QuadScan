import multiprocessing
try:
    import Queue
except ModuleNotFoundError:
    import queue as Queue

import numpy as np
import sys
import os
import re
import pprint
from tasks.GenericTasks import Task
import threading
import time
from QuadScanDataStructs import *
from PyQt5 import QtGui, QtCore, QtWidgets
from open_scan_dialog_ui import Ui_QuadFileDialog

import logging
logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class ScanDataFileSystemModel(QtWidgets.QFileSystemModel):
    def columnCount(self, parent=QtCore.QModelIndex()):
        return super(ScanDataFileSystemModel, self).columnCount() + 1

    def headerData(self, section, orientation, role):
        if section == 0:
            return super(ScanDataFileSystemModel, self).headerData(section, orientation, role)
        elif section == 1:
            return "Images"
        else:
            return super(ScanDataFileSystemModel, self).headerData(section - 1, orientation, role)

    def data(self, index, role):
        # if index.column() == self.columnCount() - 1:
        if index.column() == 0:
            return super(ScanDataFileSystemModel, self).data(index, role)
        elif index.column() == 1:
            if role == QtCore.Qt.DisplayRole:
                fileinfo = self.fileInfo(index)
                # logger.info("Fileinfo {0}".format(fileinfo.canonicalFilePath()))
                d = QtCore.QDir(fileinfo.canonicalFilePath())
                d.setNameFilters(["*daq_info*.txt"])
                ld = [str(x) for x in d.entryList()]
                # logger.debug("Found daqinfo: {0}".format(ld))
                if d.count() > 0:
                    d.setNameFilters(["*.png"])
                    # logger.debug("Found files: {0}".format(d.count()))
                    im_count = str(d.count())
                else:
                    im_count = "--"
                return im_count
            if role == QtCore.Qt.TextAlignmentRole:
                return QtCore.Qt.AlignHCenter
        else:
            idx = index.sibling(index.row(), index.column()-1)
            return super(ScanDataFileSystemModel, self).data(idx, role)


class OpenScanFileDialog(QtWidgets.QDialog):
    def __init__(self, start_dir=None, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.logger = logging.getLogger("FileDialog")
        self.logger.setLevel(logging.WARNING)

        if start_dir is None:
            start_dir = QtCore.QDir.currentPath()

        self.settings = QtCore.QSettings('Maxlab', 'QuadScanFiledialog')

        self.ui = Ui_QuadFileDialog()
        self.ui.setupUi(self)

        self.model = ScanDataFileSystemModel()
        self.model.setRootPath(QtCore.QDir.rootPath())
        self.model.directoryLoaded.connect(self.dir_loaded)
        self.model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot)
        self.logger.debug("File selection model {0}".format(self.model))

        self.target_path_index = None
        self.target_path_list = None

        self.ui.file_treeview.setModel(self.model)
        self.logger.debug("UI model {0}".format(self.ui.file_treeview.selectionModel()))
        self.ui.file_treeview.setSortingEnabled(True)
        self.ui.file_treeview.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.ui.file_treeview.setAnimated(False)
        self.ui.file_treeview.setIndentation(20)
        self.ui.file_treeview.setColumnWidth(0, self.settings.value("filename_col", 400, type=int))
        self.logger.debug("Column width: {0}".format(self.settings.value("filename_col", 400, type=int)))
        self.ui.file_treeview.selectionModel().selectionChanged.connect(self.update_selection_from_tree)

        self.logger.info("Current path: {0}".format(start_dir))
        self.target_path_list = list()
        self.expand_to_path(start_dir)

        self.ui.cancel_button.clicked.connect(self.reject)
        self.ui.select_button.clicked.connect(self.accept)
        self.ui.dir_lineedit.editingFinished.connect(self.update_selection_from_lineedit)

        # Geometry setup
        window_pos_x = self.settings.value('window_pos_x', 100, type=int)
        window_pos_y = self.settings.value('window_pos_y', 100, type=int)
        window_size_w = self.settings.value('window_size_w', 1100, type=int)
        window_size_h = self.settings.value('window_size_h', 800, type=int)
        # if window_pos_x < 50:
        #     window_pos_x = 50
        if window_pos_y < 50:
            window_pos_y = 50
        self.setGeometry(window_pos_x, window_pos_y, window_size_w, window_size_h)

        splitter_sizes = self.settings.value("splitter", [None], type="QVariantList")
        if splitter_sizes[0] is not None:
            self.ui.splitter.setSizes([np.int(s) for s in splitter_sizes])

    def closeEvent(self, event):
        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))

        self.settings.setValue("splitter", self.ui.splitter.sizes())
        self.settings.setValue("filename_col", self.ui.file_treeview.columnWidth(0))

        return super(OpenScanFileDialog, self).closeEvent(event)

    def dir_loaded(self, dir_str):
        self.logger.info("Dir {0} loaded".format(str(dir_str)))
        self.logger.info("Target path list:\n{0}".format(self.target_path_list))

        # Expand to path code:
        if self.target_path_index < len(self.target_path_list):
            tp = self.target_path_list[self.target_path_index]
            ind = self.model.index(self.target_path_list[self.target_path_index])
            self.logger.debug("Path {0} index {1} {2}".format(tp, ind.row(), ind.column()))
            self.ui.file_treeview.scrollTo(ind)
            self.ui.file_treeview.expand(ind)
            self.ui.file_treeview.entered.emit(ind)
            self.target_path_index += 1
            self.logger.debug("Selection self.model {0}, ui model {1}".format(self.model,
                                                                              self.ui.file_treeview.selectionModel()))
            if self.target_path_index < len(self.target_path_list):
                self.logger.debug("Fetch more from {0}".format(ind))
                if self.model.rowCount(ind) == 0:
                    self.model.fetchMore(ind)
                else:
                    self.dir_loaded(self.target_path_list[self.target_path_index])
            else:
                self.ui.file_treeview.selectionModel().select(ind, QtCore.QItemSelectionModel.SelectCurrent)

    def expand_to_path(self, path_name):
        self.target_path_list = str(path_name).split("/")
        self.logger.debug("Expand to {0}".format(self.target_path_list))
        self.target_path_index = 0
        for ind in range(len(self.target_path_list)-1):
            self.target_path_list[ind+1] = "{0}/{1}".format(self.target_path_list[ind], self.target_path_list[ind+1])
        self.dir_loaded(self.target_path_list[0])

    def update_selection_from_tree(self):
        self.logger.info("Selection")
        sel_ind = self.ui.file_treeview.selectionModel().selection().indexes()[0]
        row = sel_ind.row()
        sel_string = self.model.fileInfo(sel_ind).canonicalFilePath()
        parent = sel_ind.parent()
        try:
            sel_images = int(self.model.data(self.model.index(row, 1, parent), QtCore.Qt.DisplayRole))
            image_count = sel_images
        except ValueError:
            image_count = None
        # logger.info("Data: {0}, {1}, {2}".format(str(sel_string), row, sel_images))
        self.ui.dir_lineedit.setText(sel_string)
        filename = "daq_info.txt"
        load_dir_str = str(sel_string)
        if os.path.isfile(os.path.join(load_dir_str, filename)):
            self.load_daqinfo(sel_string, image_count)
        else:
            self.load_daqinfo_multi(sel_string, image_count)

    def update_selection_from_lineedit(self):
        pathname = self.ui.dir_lineedit.text()
        self.logger.info("Expanding tree to {0}".format(str(pathname)))
        self.expand_to_path(pathname)

    def directory(self):
        return QtCore.QDir(self.ui.dir_lineedit.text())

    def get_selected_path(self):
        return str(self.ui.dir_lineedit.text())

    def load_daqinfo(self, load_dir, image_count=None):
        # See if there is a file called daq_info.txt
        filename = "daq_info.txt"
        load_dir_str = str(load_dir)
        if os.path.isfile(os.path.join(load_dir_str, filename)) is False:
            e = "daq_info.txt not found in {0}".format(load_dir)
            self.logger.error(e)
            self.ui.daqinfo_label.setText("-- No data --")
            return

        self.logger.info("Loading Jason format data")
        data_dict = dict()
        with open(os.path.join(load_dir_str, filename), "r") as daq_file:
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
            num_k = int(data_dict["num_k_values"])
            num_im = int(data_dict["num_shots"])
            total_im = num_k * num_im
        except (ValueError, NameError):
            num_k = "--"
            num_im = "--"
            total_im = "--"

        try:
            s_list = load_dir_str.split("/")[-1].split("_")
        except IndexError:
            s_list[0] = "--"
            s_list[1] = "--"

        # logger.debug("Loaded data_dict: \n{0}".format(pprint.pformat(data_dict)))

        text = "Scan data from daq_info.txt:\n" \
               "==============================\n\n" \
               "\n" \
               "Quad         {0}\n" \
               "Num k pos    {1}\n" \
               "\n" \
               "Screen       {2}\n" \
               "Images/pos   {3}\n" \
               "\n" \
               "Total images {4}\n" \
               "Images found {5}\n" \
               "\n" \
               "Energy       {6} MeV\n" \
               "k_min        {7:.2f} \n" \
               "k_max        {8:.2f} \n" \
               "\n" \
               "Date         {9}\n" \
               "Time         {10}".format(data_dict["quad"],
                                          num_k,
                                          data_dict["screen"],
                                          num_im,
                                          total_im,
                                          image_count,
                                          data_dict["beam_energy"],
                                          float(data_dict["k_min"]),
                                          float(data_dict["k_max"]),
                                          s_list[0],
                                          s_list[1].replace("-", ":"))
        self.ui.daqinfo_label.setText(text)

    def load_daqinfo_multi(self, load_dir, image_count=None):
        # See if there is a file called daq_info.txt
        filename = "daq_info_multi.txt"
        load_dir_str = str(load_dir)
        if os.path.isfile(os.path.join(load_dir_str, filename)) is False:
            e = "daq_info_multi.txt not found in {0}".format(load_dir)
            self.logger.error(e)
            self.ui.daqinfo_label.setText("-- No data --")
            return

        self.logger.info("Loading Jason format data")
        data_dict = dict()
        n_quads = 0
        with open(os.path.join(load_dir_str, filename), "r") as daq_file:
            while True:
                line = daq_file.readline()
                if line == "" or line[0:5] == "*****":
                    break
                try:
                    key, value = line.split(":")
                    # self.logger.info("Found key: {0}, match {1}".format(key, m))
                    k = key.strip()
                    data_dict[k] = value.strip()
                    m = re.match("quad_[0-9]*$", k)
                    if m:
                        n_quads += 1
                except ValueError:
                    pass

        try:
            num_k = int(data_dict["num_k_values"])
            num_im = int(data_dict["num_shots"])
            total_im = num_k * num_im
        except (ValueError, NameError):
            num_k = "--"
            num_im = "--"
            total_im = "--"

        try:
            s_list = load_dir_str.split("/")[-1].split("_")
        except IndexError:
            s_list[0] = "--"
            s_list[1] = "--"

        # logger.debug("Loaded data_dict: \n{0}".format(pprint.pformat(data_dict)))

        text = "Scan data from daq_info_multi.txt:\n" \
               "===================================\n\n" \
               "\n" \
               "Quads        {0}\n" \
               "Num k pos    {1}\n" \
               "\n" \
               "Screen       {2}\n" \
               "Images/pos   {3}\n" \
               "\n" \
               "Total images {4}\n" \
               "Images found {5}\n" \
               "\n" \
               "Energy       {6} MeV\n" \
               "k_min        {7:.2f} \n" \
               "k_max        {8:.2f} \n" \
               "\n" \
               "Date         {9}\n" \
               "Time         {10}".format(n_quads,
                                          num_k,
                                          data_dict["screen"],
                                          num_im,
                                          total_im,
                                          image_count,
                                          data_dict["beam_energy"],
                                          float(data_dict["k_min"]),
                                          float(data_dict["k_max"]),
                                          s_list[0],
                                          s_list[1].replace("-", ":"))
        self.ui.daqinfo_label.setText(text)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = OpenScanFileDialog("D:/Programming/emittancesinglequad/saved-images")
    myapp.show()
    sys.exit(app.exec_())

