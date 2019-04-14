import multiprocessing
try:
    import Queue
except ModuleNotFoundError:
    import queue as Queue

import numpy as np
from tasks.GenericTasks import Task
import threading
import time
from QuadScanDataStructs import *
from PyQt4 import QtGui, QtCore
from open_scan_dialog_ui import Ui_QuadFileDialog

import logging
logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


class MyFileSystemModel(QtGui.QFileSystemModel):
    def columnCount(self, parent=QtCore.QModelIndex()):
        return super(MyFileSystemModel, self).columnCount() + 1

    def data(self, index, role):
        if index.column() == self.columnCount() - 1:
            if role == QtCore.Qt.DisplayRole:
                return QtCore.QString("YourText")
            if role == QtCore.Qt.TextAlignmentRole:
                return QtCore.Qt.AlignHCenter

        return super(MyFileSystemModel, self).data(index, role)


class MyFileDailog(QtGui.QWidget):
    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)

        self.ui = Ui_QuadFileDialog()
        self.ui.setupUi(self)


if __name__ == "__main__":
    fd = MyFileDailog()
    lay = fd.layout()
    spl = lay.itemAt(2).widget()        # splitter
