# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ab_ellipse_tester_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1056, 598)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, -1, 6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.alpha_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.alpha_spinbox.setMinimumSize(QtCore.QSize(70, 0))
        self.alpha_spinbox.setMinimum(-10000.0)
        self.alpha_spinbox.setMaximum(10000.0)
        self.alpha_spinbox.setObjectName("alpha_spinbox")
        self.gridLayout.addWidget(self.alpha_spinbox, 0, 1, 1, 1)
        self.screen_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.screen_max_spinbox.setProperty("value", 2.0)
        self.screen_max_spinbox.setObjectName("screen_max_spinbox")
        self.gridLayout.addWidget(self.screen_max_spinbox, 10, 4, 1, 1)
        self.k1_slider = QtWidgets.QSlider(Dialog)
        self.k1_slider.setProperty("value", 50)
        self.k1_slider.setOrientation(QtCore.Qt.Horizontal)
        self.k1_slider.setObjectName("k1_slider")
        self.gridLayout.addWidget(self.k1_slider, 6, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.k1_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k1_max_spinbox.setMinimum(-10.0)
        self.k1_max_spinbox.setMaximum(10.0)
        self.k1_max_spinbox.setProperty("value", 5.0)
        self.k1_max_spinbox.setObjectName("k1_max_spinbox")
        self.gridLayout.addWidget(self.k1_max_spinbox, 6, 4, 1, 1)
        self.k4_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k4_max_spinbox.setMinimum(-10.0)
        self.k4_max_spinbox.setMaximum(10.0)
        self.k4_max_spinbox.setProperty("value", 5.0)
        self.k4_max_spinbox.setObjectName("k4_max_spinbox")
        self.gridLayout.addWidget(self.k4_max_spinbox, 9, 4, 1, 1)
        self.k4_slider = QtWidgets.QSlider(Dialog)
        self.k4_slider.setProperty("value", 50)
        self.k4_slider.setOrientation(QtCore.Qt.Horizontal)
        self.k4_slider.setObjectName("k4_slider")
        self.gridLayout.addWidget(self.k4_slider, 9, 3, 1, 1)
        self.alpha_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.alpha_max_spinbox.setMinimum(-10000.0)
        self.alpha_max_spinbox.setMaximum(10000.0)
        self.alpha_max_spinbox.setProperty("value", 50.0)
        self.alpha_max_spinbox.setObjectName("alpha_max_spinbox")
        self.gridLayout.addWidget(self.alpha_max_spinbox, 0, 4, 1, 1)
        self.alpha_slider = QtWidgets.QSlider(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.alpha_slider.sizePolicy().hasHeightForWidth())
        self.alpha_slider.setSizePolicy(sizePolicy)
        self.alpha_slider.setMinimumSize(QtCore.QSize(200, 0))
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(99)
        self.alpha_slider.setProperty("value", 50)
        self.alpha_slider.setOrientation(QtCore.Qt.Horizontal)
        self.alpha_slider.setObjectName("alpha_slider")
        self.gridLayout.addWidget(self.alpha_slider, 0, 3, 1, 1)
        self.eps_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.eps_max_spinbox.setProperty("value", 10.0)
        self.eps_max_spinbox.setObjectName("eps_max_spinbox")
        self.gridLayout.addWidget(self.eps_max_spinbox, 2, 4, 1, 1)
        self.k3_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k3_max_spinbox.setMinimum(-10.0)
        self.k3_max_spinbox.setMaximum(10.0)
        self.k3_max_spinbox.setProperty("value", 5.0)
        self.k3_max_spinbox.setObjectName("k3_max_spinbox")
        self.gridLayout.addWidget(self.k3_max_spinbox, 8, 4, 1, 1)
        self.k2_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k2_max_spinbox.setMinimum(-10.0)
        self.k2_max_spinbox.setMaximum(10.0)
        self.k2_max_spinbox.setProperty("value", 5.0)
        self.k2_max_spinbox.setObjectName("k2_max_spinbox")
        self.gridLayout.addWidget(self.k2_max_spinbox, 7, 4, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.sigma_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.sigma_max_spinbox.setProperty("value", 2.0)
        self.sigma_max_spinbox.setObjectName("sigma_max_spinbox")
        self.gridLayout.addWidget(self.sigma_max_spinbox, 3, 4, 1, 1)
        self.beta_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.beta_min_spinbox.setMaximum(10000.0)
        self.beta_min_spinbox.setObjectName("beta_min_spinbox")
        self.gridLayout.addWidget(self.beta_min_spinbox, 1, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 10, 0, 1, 1)
        self.alpha_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.alpha_min_spinbox.setMinimum(-10000.0)
        self.alpha_min_spinbox.setMaximum(10000.0)
        self.alpha_min_spinbox.setProperty("value", -50.0)
        self.alpha_min_spinbox.setObjectName("alpha_min_spinbox")
        self.gridLayout.addWidget(self.alpha_min_spinbox, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.k2_slider = QtWidgets.QSlider(Dialog)
        self.k2_slider.setProperty("value", 50)
        self.k2_slider.setOrientation(QtCore.Qt.Horizontal)
        self.k2_slider.setObjectName("k2_slider")
        self.gridLayout.addWidget(self.k2_slider, 7, 3, 1, 1)
        self.sigma_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.sigma_min_spinbox.setObjectName("sigma_min_spinbox")
        self.gridLayout.addWidget(self.sigma_min_spinbox, 3, 2, 1, 1)
        self.energy_slider = QtWidgets.QSlider(Dialog)
        self.energy_slider.setProperty("value", 25)
        self.energy_slider.setOrientation(QtCore.Qt.Horizontal)
        self.energy_slider.setObjectName("energy_slider")
        self.gridLayout.addWidget(self.energy_slider, 4, 3, 1, 1)
        self.beta_slider = QtWidgets.QSlider(Dialog)
        self.beta_slider.setMaximum(99)
        self.beta_slider.setProperty("value", 50)
        self.beta_slider.setOrientation(QtCore.Qt.Horizontal)
        self.beta_slider.setObjectName("beta_slider")
        self.gridLayout.addWidget(self.beta_slider, 1, 3, 1, 1)
        self.k2_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k2_min_spinbox.setMinimum(-10.0)
        self.k2_min_spinbox.setMaximum(10.0)
        self.k2_min_spinbox.setProperty("value", -5.0)
        self.k2_min_spinbox.setObjectName("k2_min_spinbox")
        self.gridLayout.addWidget(self.k2_min_spinbox, 7, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 5, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 7, 0, 1, 1)
        self.k4_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k4_min_spinbox.setMinimum(-10.0)
        self.k4_min_spinbox.setMaximum(10.0)
        self.k4_min_spinbox.setProperty("value", -5.0)
        self.k4_min_spinbox.setObjectName("k4_min_spinbox")
        self.gridLayout.addWidget(self.k4_min_spinbox, 9, 2, 1, 1)
        self.sigma_slider = QtWidgets.QSlider(Dialog)
        self.sigma_slider.setProperty("value", 50)
        self.sigma_slider.setOrientation(QtCore.Qt.Horizontal)
        self.sigma_slider.setObjectName("sigma_slider")
        self.gridLayout.addWidget(self.sigma_slider, 3, 3, 1, 1)
        self.screen_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.screen_min_spinbox.setObjectName("screen_min_spinbox")
        self.gridLayout.addWidget(self.screen_min_spinbox, 10, 2, 1, 1)
        self.screen_slider = QtWidgets.QSlider(Dialog)
        self.screen_slider.setProperty("value", 19)
        self.screen_slider.setOrientation(QtCore.Qt.Horizontal)
        self.screen_slider.setObjectName("screen_slider")
        self.gridLayout.addWidget(self.screen_slider, 10, 3, 1, 1)
        self.k3_slider = QtWidgets.QSlider(Dialog)
        self.k3_slider.setProperty("value", 50)
        self.k3_slider.setOrientation(QtCore.Qt.Horizontal)
        self.k3_slider.setObjectName("k3_slider")
        self.gridLayout.addWidget(self.k3_slider, 8, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 6, 0, 1, 1)
        self.k3_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k3_min_spinbox.setMinimum(-10.0)
        self.k3_min_spinbox.setMaximum(10.0)
        self.k3_min_spinbox.setProperty("value", -5.0)
        self.k3_min_spinbox.setObjectName("k3_min_spinbox")
        self.gridLayout.addWidget(self.k3_min_spinbox, 8, 2, 1, 1)
        self.eps_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.eps_min_spinbox.setMinimum(0.1)
        self.eps_min_spinbox.setObjectName("eps_min_spinbox")
        self.gridLayout.addWidget(self.eps_min_spinbox, 2, 2, 1, 1)
        self.energy_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.energy_min_spinbox.setMinimum(1.0)
        self.energy_min_spinbox.setObjectName("energy_min_spinbox")
        self.gridLayout.addWidget(self.energy_min_spinbox, 4, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 8, 0, 1, 1)
        self.k1_min_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k1_min_spinbox.setMinimum(-10.0)
        self.k1_min_spinbox.setMaximum(10.0)
        self.k1_min_spinbox.setProperty("value", -5.0)
        self.k1_min_spinbox.setObjectName("k1_min_spinbox")
        self.gridLayout.addWidget(self.k1_min_spinbox, 6, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 11, 0, 1, 1)
        self.energy_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.energy_max_spinbox.setMaximum(1000.0)
        self.energy_max_spinbox.setProperty("value", 1000.0)
        self.energy_max_spinbox.setObjectName("energy_max_spinbox")
        self.gridLayout.addWidget(self.energy_max_spinbox, 4, 4, 1, 1)
        self.beta_max_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.beta_max_spinbox.setMaximum(10000.0)
        self.beta_max_spinbox.setProperty("value", 100.0)
        self.beta_max_spinbox.setObjectName("beta_max_spinbox")
        self.gridLayout.addWidget(self.beta_max_spinbox, 1, 4, 1, 1)
        self.eps_slider = QtWidgets.QSlider(Dialog)
        self.eps_slider.setMaximum(99)
        self.eps_slider.setProperty("value", 30)
        self.eps_slider.setOrientation(QtCore.Qt.Horizontal)
        self.eps_slider.setObjectName("eps_slider")
        self.gridLayout.addWidget(self.eps_slider, 2, 3, 1, 1)
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 12, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 9, 0, 1, 1)
        self.k1_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k1_spinbox.setMinimum(-10.0)
        self.k1_spinbox.setMaximum(10.0)
        self.k1_spinbox.setObjectName("k1_spinbox")
        self.gridLayout.addWidget(self.k1_spinbox, 6, 1, 1, 1)
        self.k3_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k3_spinbox.setMinimum(-10.0)
        self.k3_spinbox.setMaximum(10.0)
        self.k3_spinbox.setObjectName("k3_spinbox")
        self.gridLayout.addWidget(self.k3_spinbox, 8, 1, 1, 1)
        self.k4_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k4_spinbox.setMinimum(-10.0)
        self.k4_spinbox.setMaximum(10.0)
        self.k4_spinbox.setObjectName("k4_spinbox")
        self.gridLayout.addWidget(self.k4_spinbox, 9, 1, 1, 1)
        self.sigma_x_label = QtWidgets.QLabel(Dialog)
        self.sigma_x_label.setObjectName("sigma_x_label")
        self.gridLayout.addWidget(self.sigma_x_label, 12, 1, 1, 1)
        self.beta_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.beta_spinbox.setMaximum(10000.0)
        self.beta_spinbox.setObjectName("beta_spinbox")
        self.gridLayout.addWidget(self.beta_spinbox, 1, 1, 1, 1)
        self.k2_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.k2_spinbox.setMinimum(-10.0)
        self.k2_spinbox.setMaximum(10.0)
        self.k2_spinbox.setObjectName("k2_spinbox")
        self.gridLayout.addWidget(self.k2_spinbox, 7, 1, 1, 1)
        self.eps_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.eps_spinbox.setMaximum(100.0)
        self.eps_spinbox.setObjectName("eps_spinbox")
        self.gridLayout.addWidget(self.eps_spinbox, 2, 1, 1, 1)
        self.screen_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.screen_spinbox.setProperty("value", 0.0)
        self.screen_spinbox.setObjectName("screen_spinbox")
        self.gridLayout.addWidget(self.screen_spinbox, 10, 1, 1, 1)
        self.energy_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.energy_spinbox.setMaximum(10000.0)
        self.energy_spinbox.setObjectName("energy_spinbox")
        self.gridLayout.addWidget(self.energy_spinbox, 4, 1, 1, 1)
        self.sigma_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.sigma_spinbox.setObjectName("sigma_spinbox")
        self.gridLayout.addWidget(self.sigma_spinbox, 3, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 13, 0, 1, 1)
        self.sigma_y_label = QtWidgets.QLabel(Dialog)
        self.sigma_y_label.setObjectName("sigma_y_label")
        self.gridLayout.addWidget(self.sigma_y_label, 13, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.widget = PlotWidget(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.horizontalLayout.addWidget(self.widget)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.alpha_spinbox, self.beta_spinbox)
        Dialog.setTabOrder(self.beta_spinbox, self.eps_spinbox)
        Dialog.setTabOrder(self.eps_spinbox, self.sigma_spinbox)
        Dialog.setTabOrder(self.sigma_spinbox, self.energy_spinbox)
        Dialog.setTabOrder(self.energy_spinbox, self.alpha_slider)
        Dialog.setTabOrder(self.alpha_slider, self.beta_slider)
        Dialog.setTabOrder(self.beta_slider, self.eps_slider)
        Dialog.setTabOrder(self.eps_slider, self.sigma_slider)
        Dialog.setTabOrder(self.sigma_slider, self.energy_slider)
        Dialog.setTabOrder(self.energy_slider, self.k1_spinbox)
        Dialog.setTabOrder(self.k1_spinbox, self.k2_spinbox)
        Dialog.setTabOrder(self.k2_spinbox, self.k3_spinbox)
        Dialog.setTabOrder(self.k3_spinbox, self.k4_spinbox)
        Dialog.setTabOrder(self.k4_spinbox, self.screen_spinbox)
        Dialog.setTabOrder(self.screen_spinbox, self.k1_slider)
        Dialog.setTabOrder(self.k1_slider, self.k2_slider)
        Dialog.setTabOrder(self.k2_slider, self.k3_slider)
        Dialog.setTabOrder(self.k3_slider, self.k4_slider)
        Dialog.setTabOrder(self.k4_slider, self.screen_slider)
        Dialog.setTabOrder(self.screen_slider, self.alpha_min_spinbox)
        Dialog.setTabOrder(self.alpha_min_spinbox, self.alpha_max_spinbox)
        Dialog.setTabOrder(self.alpha_max_spinbox, self.beta_min_spinbox)
        Dialog.setTabOrder(self.beta_min_spinbox, self.beta_max_spinbox)
        Dialog.setTabOrder(self.beta_max_spinbox, self.eps_min_spinbox)
        Dialog.setTabOrder(self.eps_min_spinbox, self.eps_max_spinbox)
        Dialog.setTabOrder(self.eps_max_spinbox, self.sigma_min_spinbox)
        Dialog.setTabOrder(self.sigma_min_spinbox, self.sigma_max_spinbox)
        Dialog.setTabOrder(self.sigma_max_spinbox, self.energy_min_spinbox)
        Dialog.setTabOrder(self.energy_min_spinbox, self.energy_max_spinbox)
        Dialog.setTabOrder(self.energy_max_spinbox, self.k1_min_spinbox)
        Dialog.setTabOrder(self.k1_min_spinbox, self.k1_max_spinbox)
        Dialog.setTabOrder(self.k1_max_spinbox, self.k2_min_spinbox)
        Dialog.setTabOrder(self.k2_min_spinbox, self.k2_max_spinbox)
        Dialog.setTabOrder(self.k2_max_spinbox, self.k3_min_spinbox)
        Dialog.setTabOrder(self.k3_min_spinbox, self.k3_max_spinbox)
        Dialog.setTabOrder(self.k3_max_spinbox, self.k4_min_spinbox)
        Dialog.setTabOrder(self.k4_min_spinbox, self.k4_max_spinbox)
        Dialog.setTabOrder(self.k4_max_spinbox, self.screen_min_spinbox)
        Dialog.setTabOrder(self.screen_min_spinbox, self.screen_max_spinbox)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_3.setText(_translate("Dialog", "eps / umrad"))
        self.label.setText(_translate("Dialog", "alpha"))
        self.label_5.setText(_translate("Dialog", "energy / MeV"))
        self.label_2.setText(_translate("Dialog", "beta / m"))
        self.label_10.setText(_translate("Dialog", "screen pos"))
        self.label_4.setText(_translate("Dialog", "sigma / mm"))
        self.label_9.setText(_translate("Dialog", "k2"))
        self.label_6.setText(_translate("Dialog", "k1"))
        self.label_8.setText(_translate("Dialog", "k3"))
        self.label_12.setText(_translate("Dialog", "sigma_x"))
        self.label_7.setText(_translate("Dialog", "k4"))
        self.sigma_x_label.setText(_translate("Dialog", "-.--"))
        self.label_13.setText(_translate("Dialog", "sigma_y"))
        self.sigma_y_label.setText(_translate("Dialog", "-.--"))

from pyqtgraph import PlotWidget