"""
Created: 2018-10-12

Test delay in init method of gui when launhing thread

@author: Filip Lindau
"""

import logging
import threading
import sys
import time
from PyQt5 import QtGui, QtCore, QtWidgets

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.DEBUG)


class TestGui(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.label = None
        self.layout = None
        self.setup_layout()
        root.info("Creating thread")
        self.t = threading.Thread(target=self.target)
        root.info("Starting thread")
        self.t.start()
        root.info("Exiting init")

    def target(self):
        root.info("Entering target. Block for 2 s")
        t0 = time.time()
        counter = 0
        while time.time() - t0 < 2:
            counter += 1
        root.info("Exiting target. Counter: {0}".format(counter))

    def setup_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("yo yo yo")
        self.layout.addWidget(self.label)
        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = TestGui()
    root.info("Gui object created")
    myapp.show()
    root.info("App show")
    sys.exit(app.exec_())
    root.info("App exit")
