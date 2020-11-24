# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\open_scan_dialog_ui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_QuadFileDialog(object):
    def setupUi(self, QuadFileDialog):
        QuadFileDialog.setObjectName("QuadFileDialog")
        QuadFileDialog.resize(437, 359)
        self.verticalLayout = QtWidgets.QVBoxLayout(QuadFileDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(QuadFileDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.file_treeview = QtWidgets.QTreeView(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_treeview.sizePolicy().hasHeightForWidth())
        self.file_treeview.setSizePolicy(sizePolicy)
        self.file_treeview.setObjectName("file_treeview")
        self.daqinfo_label = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.daqinfo_label.sizePolicy().hasHeightForWidth())
        self.daqinfo_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.daqinfo_label.setFont(font)
        self.daqinfo_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.daqinfo_label.setObjectName("daqinfo_label")
        self.horizontalLayout_3.addWidget(self.splitter)
        self.verticalLayout.addWidget(self.widget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 6, -1, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(QuadFileDialog)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.dir_lineedit = QtWidgets.QLineEdit(QuadFileDialog)
        self.dir_lineedit.setObjectName("dir_lineedit")
        self.horizontalLayout_2.addWidget(self.dir_lineedit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.select_button = QtWidgets.QPushButton(QuadFileDialog)
        self.select_button.setObjectName("select_button")
        self.gridLayout.addWidget(self.select_button, 0, 1, 1, 1)
        self.cancel_button = QtWidgets.QPushButton(QuadFileDialog)
        self.cancel_button.setObjectName("cancel_button")
        self.gridLayout.addWidget(self.cancel_button, 1, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(QuadFileDialog)
        QtCore.QMetaObject.connectSlotsByName(QuadFileDialog)

    def retranslateUi(self, QuadFileDialog):
        _translate = QtCore.QCoreApplication.translate
        QuadFileDialog.setWindowTitle(_translate("QuadFileDialog", "Open scan"))
        self.daqinfo_label.setText(_translate("QuadFileDialog", "TextLabel"))
        self.label_2.setText(_translate("QuadFileDialog", "Dir name"))
        self.select_button.setText(_translate("QuadFileDialog", "Select"))
        self.cancel_button.setText(_translate("QuadFileDialog", "Cancel"))
