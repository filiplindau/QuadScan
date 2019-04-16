# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open_scan_dialog_ui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_QuadFileDialog(object):
    def setupUi(self, QuadFileDialog):
        QuadFileDialog.setObjectName(_fromUtf8("QuadFileDialog"))
        QuadFileDialog.resize(437, 359)
        self.verticalLayout = QtGui.QVBoxLayout(QuadFileDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.widget = QtGui.QWidget(QuadFileDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.splitter = QtGui.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.file_treeview = QtGui.QTreeView(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_treeview.sizePolicy().hasHeightForWidth())
        self.file_treeview.setSizePolicy(sizePolicy)
        self.file_treeview.setObjectName(_fromUtf8("file_treeview"))
        self.daqinfo_label = QtGui.QLabel(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.daqinfo_label.sizePolicy().hasHeightForWidth())
        self.daqinfo_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Consolas"))
        self.daqinfo_label.setFont(font)
        self.daqinfo_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.daqinfo_label.setObjectName(_fromUtf8("daqinfo_label"))
        self.horizontalLayout_3.addWidget(self.splitter)
        self.verticalLayout.addWidget(self.widget)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 6, -1, -1)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(QuadFileDialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.dir_lineedit = QtGui.QLineEdit(QuadFileDialog)
        self.dir_lineedit.setObjectName(_fromUtf8("dir_lineedit"))
        self.horizontalLayout_2.addWidget(self.dir_lineedit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.select_button = QtGui.QPushButton(QuadFileDialog)
        self.select_button.setObjectName(_fromUtf8("select_button"))
        self.gridLayout.addWidget(self.select_button, 0, 1, 1, 1)
        self.cancel_button = QtGui.QPushButton(QuadFileDialog)
        self.cancel_button.setObjectName(_fromUtf8("cancel_button"))
        self.gridLayout.addWidget(self.cancel_button, 1, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(QuadFileDialog)
        QtCore.QMetaObject.connectSlotsByName(QuadFileDialog)

    def retranslateUi(self, QuadFileDialog):
        QuadFileDialog.setWindowTitle(_translate("QuadFileDialog", "Open scan", None))
        self.daqinfo_label.setText(_translate("QuadFileDialog", "TextLabel", None))
        self.label_2.setText(_translate("QuadFileDialog", "Dir name", None))
        self.select_button.setText(_translate("QuadFileDialog", "Select", None))
        self.cancel_button.setText(_translate("QuadFileDialog", "Cancel", None))

