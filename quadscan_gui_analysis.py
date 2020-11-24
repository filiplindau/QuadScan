# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\quadscan_gui_analysis.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_QuadScanDialog(object):
    def setupUi(self, QuadScanDialog):
        QuadScanDialog.setObjectName("QuadScanDialog")
        QuadScanDialog.resize(1121, 526)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/eps.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        QuadScanDialog.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(QuadScanDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        spacerItem = QtWidgets.QSpacerItem(3, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem)
        self.load_disk_button = QtWidgets.QPushButton(QuadScanDialog)
        self.load_disk_button.setMaximumSize(QtCore.QSize(60, 16777215))
        self.load_disk_button.setObjectName("load_disk_button")
        self.horizontalLayout_14.addWidget(self.load_disk_button)
        spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem1)
        self.data_source_label = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.data_source_label.sizePolicy().hasHeightForWidth())
        self.data_source_label.setSizePolicy(sizePolicy)
        self.data_source_label.setObjectName("data_source_label")
        self.horizontalLayout_14.addWidget(self.data_source_label)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setContentsMargins(-1, -1, 6, -1)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_33 = QtWidgets.QLabel(QuadScanDialog)
        self.label_33.setObjectName("label_33")
        self.gridLayout_3.addWidget(self.label_33, 18, 0, 1, 1)
        self.p_image_index_slider = QtWidgets.QSlider(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.p_image_index_slider.sizePolicy().hasHeightForWidth())
        self.p_image_index_slider.setSizePolicy(sizePolicy)
        self.p_image_index_slider.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.p_image_index_slider.setMaximum(10)
        self.p_image_index_slider.setOrientation(QtCore.Qt.Horizontal)
        self.p_image_index_slider.setInvertedControls(False)
        self.p_image_index_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.p_image_index_slider.setObjectName("p_image_index_slider")
        self.gridLayout_3.addWidget(self.p_image_index_slider, 6, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy)
        self.label_30.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_30.setObjectName("label_30")
        self.gridLayout_3.addWidget(self.label_30, 17, 0, 1, 1)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.p_roi_size_w_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_roi_size_w_spinbox.setMaximum(5000.0)
        self.p_roi_size_w_spinbox.setObjectName("p_roi_size_w_spinbox")
        self.horizontalLayout_13.addWidget(self.p_roi_size_w_spinbox)
        self.p_roi_size_h_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_roi_size_h_spinbox.setMaximum(5000.0)
        self.p_roi_size_h_spinbox.setObjectName("p_roi_size_h_spinbox")
        self.horizontalLayout_13.addWidget(self.p_roi_size_h_spinbox)
        self.gridLayout_3.addLayout(self.horizontalLayout_13, 11, 1, 1, 1)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.p_roi_cent_x_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_roi_cent_x_spinbox.setMaximum(5000.0)
        self.p_roi_cent_x_spinbox.setObjectName("p_roi_cent_x_spinbox")
        self.horizontalLayout_12.addWidget(self.p_roi_cent_x_spinbox)
        self.p_roi_cent_y_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_roi_cent_y_spinbox.setMaximum(5000.0)
        self.p_roi_cent_y_spinbox.setObjectName("p_roi_cent_y_spinbox")
        self.horizontalLayout_12.addWidget(self.p_roi_cent_y_spinbox)
        self.gridLayout_3.addLayout(self.horizontalLayout_12, 10, 1, 1, 1)
        self.label_27 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy)
        self.label_27.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_27.setObjectName("label_27")
        self.gridLayout_3.addWidget(self.label_27, 11, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2, 8, 0, 1, 1)
        self.p_quad_screen_dist_label = QtWidgets.QLabel(QuadScanDialog)
        self.p_quad_screen_dist_label.setObjectName("p_quad_screen_dist_label")
        self.gridLayout_3.addWidget(self.p_quad_screen_dist_label, 3, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_24.setObjectName("label_24")
        self.gridLayout_3.addWidget(self.label_24, 3, 0, 1, 1)
        self.p_electron_energy_label = QtWidgets.QLabel(QuadScanDialog)
        self.p_electron_energy_label.setObjectName("p_electron_energy_label")
        self.gridLayout_3.addWidget(self.p_electron_energy_label, 4, 1, 1, 1)
        self.fit_algo_combobox = QtWidgets.QComboBox(QuadScanDialog)
        self.fit_algo_combobox.setObjectName("fit_algo_combobox")
        self.gridLayout_3.addWidget(self.fit_algo_combobox, 17, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_29.sizePolicy().hasHeightForWidth())
        self.label_29.setSizePolicy(sizePolicy)
        self.label_29.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_29.setObjectName("label_29")
        self.gridLayout_3.addWidget(self.label_29, 16, 0, 1, 1)
        self.p_quad_length_label = QtWidgets.QLabel(QuadScanDialog)
        self.p_quad_length_label.setObjectName("p_quad_length_label")
        self.gridLayout_3.addWidget(self.p_quad_length_label, 2, 1, 1, 1)
        self.label_28 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy)
        self.label_28.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_28.setObjectName("label_28")
        self.gridLayout_3.addWidget(self.label_28, 14, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_26.sizePolicy().hasHeightForWidth())
        self.label_26.setSizePolicy(sizePolicy)
        self.label_26.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_26.setObjectName("label_26")
        self.gridLayout_3.addWidget(self.label_26, 10, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy)
        self.label_25.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_25.setObjectName("label_25")
        self.gridLayout_3.addWidget(self.label_25, 4, 0, 1, 1)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.p_x_radio = QtWidgets.QRadioButton(QuadScanDialog)
        self.p_x_radio.setChecked(True)
        self.p_x_radio.setObjectName("p_x_radio")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(QuadScanDialog)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.p_x_radio)
        self.horizontalLayout_17.addWidget(self.p_x_radio)
        self.p_y_radio = QtWidgets.QRadioButton(QuadScanDialog)
        self.p_y_radio.setObjectName("p_y_radio")
        self.buttonGroup_2.addButton(self.p_y_radio)
        self.horizontalLayout_17.addWidget(self.p_y_radio)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem3)
        self.gridLayout_3.addLayout(self.horizontalLayout_17, 18, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_3.addItem(spacerItem4, 5, 0, 1, 1)
        self.label_32 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_32.sizePolicy().hasHeightForWidth())
        self.label_32.setSizePolicy(sizePolicy)
        self.label_32.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_32.setObjectName("label_32")
        self.gridLayout_3.addWidget(self.label_32, 6, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_23.setObjectName("label_23")
        self.gridLayout_3.addWidget(self.label_23, 2, 0, 1, 1)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.p_raw_image_radio = QtWidgets.QRadioButton(QuadScanDialog)
        self.p_raw_image_radio.setObjectName("p_raw_image_radio")
        self.buttonGroup = QtWidgets.QButtonGroup(QuadScanDialog)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.p_raw_image_radio)
        self.horizontalLayout_15.addWidget(self.p_raw_image_radio)
        self.p_filtered_image_radio = QtWidgets.QRadioButton(QuadScanDialog)
        self.p_filtered_image_radio.setChecked(True)
        self.p_filtered_image_radio.setObjectName("p_filtered_image_radio")
        self.buttonGroup.addButton(self.p_filtered_image_radio)
        self.horizontalLayout_15.addWidget(self.p_filtered_image_radio)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem5)
        self.gridLayout_3.addLayout(self.horizontalLayout_15, 7, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(QuadScanDialog)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 7, 0, 1, 1)
        self.p_threshold_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_threshold_spinbox.setMaximum(66000.0)
        self.p_threshold_spinbox.setObjectName("p_threshold_spinbox")
        self.gridLayout_3.addWidget(self.p_threshold_spinbox, 14, 1, 1, 1)
        self.p_median_kernel_spinbox = QtWidgets.QSpinBox(QuadScanDialog)
        self.p_median_kernel_spinbox.setMinimum(1)
        self.p_median_kernel_spinbox.setMaximum(21)
        self.p_median_kernel_spinbox.setSingleStep(2)
        self.p_median_kernel_spinbox.setProperty("value", 3)
        self.p_median_kernel_spinbox.setObjectName("p_median_kernel_spinbox")
        self.gridLayout_3.addWidget(self.p_median_kernel_spinbox, 16, 1, 1, 1)
        self.p_keep_charge_ratio_spinbox = QtWidgets.QDoubleSpinBox(QuadScanDialog)
        self.p_keep_charge_ratio_spinbox.setMaximum(100.0)
        self.p_keep_charge_ratio_spinbox.setProperty("value", 100.0)
        self.p_keep_charge_ratio_spinbox.setObjectName("p_keep_charge_ratio_spinbox")
        self.gridLayout_3.addWidget(self.p_keep_charge_ratio_spinbox, 15, 1, 1, 1)
        self.label_31 = QtWidgets.QLabel(QuadScanDialog)
        self.label_31.setObjectName("label_31")
        self.gridLayout_3.addWidget(self.label_31, 15, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 0, 6, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.process_button = QtWidgets.QPushButton(QuadScanDialog)
        self.process_button.setMaximumSize(QtCore.QSize(75, 16777215))
        self.process_button.setObjectName("process_button")
        self.horizontalLayout.addWidget(self.process_button)
        self.init_button = QtWidgets.QPushButton(QuadScanDialog)
        self.init_button.setMaximumSize(QtCore.QSize(75, 16777215))
        self.init_button.setObjectName("init_button")
        self.horizontalLayout.addWidget(self.init_button)
        self.gridLayout_3.addLayout(self.horizontalLayout, 20, 1, 1, 1)
        self.horizontalLayout_16.addLayout(self.gridLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_3 = QtWidgets.QWidget(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_4.setContentsMargins(-1, 3, -1, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.analysis_pic_plot_splitter = QtWidgets.QSplitter(self.widget_3)
        self.analysis_pic_plot_splitter.setStyleSheet("QSplitter::handle {\n"
"    image:url(:/icons/hor_splitter_handle.png);\n"
"    width: 15px;\n"
"    height: 10px;\n"
"}")
        self.analysis_pic_plot_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.analysis_pic_plot_splitter.setObjectName("analysis_pic_plot_splitter")
        self.widget_7 = QtWidgets.QWidget(self.analysis_pic_plot_splitter)
        self.widget_7.setObjectName("widget_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setContentsMargins(-1, 0, 0, -1)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.line_10 = QtWidgets.QFrame(self.widget_7)
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.horizontalLayout_11.addWidget(self.line_10)
        self.p_k_value_label = QtWidgets.QLabel(self.widget_7)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.p_k_value_label.setFont(font)
        self.p_k_value_label.setObjectName("p_k_value_label")
        self.horizontalLayout_11.addWidget(self.p_k_value_label)
        self.line_8 = QtWidgets.QFrame(self.widget_7)
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.horizontalLayout_11.addWidget(self.line_8)
        self.p_k_ind_label = QtWidgets.QLabel(self.widget_7)
        self.p_k_ind_label.setObjectName("p_k_ind_label")
        self.horizontalLayout_11.addWidget(self.p_k_ind_label)
        self.line_9 = QtWidgets.QFrame(self.widget_7)
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.horizontalLayout_11.addWidget(self.line_9)
        self.p_image_label = QtWidgets.QLabel(self.widget_7)
        self.p_image_label.setObjectName("p_image_label")
        self.horizontalLayout_11.addWidget(self.p_image_label)
        self.line_11 = QtWidgets.QFrame(self.widget_7)
        self.line_11.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.horizontalLayout_11.addWidget(self.line_11)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem6)
        self.verticalLayout_6.addLayout(self.horizontalLayout_11)
        self.process_image_widget = ImageView(self.widget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.process_image_widget.sizePolicy().hasHeightForWidth())
        self.process_image_widget.setSizePolicy(sizePolicy)
        self.process_image_widget.setObjectName("process_image_widget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.process_image_widget)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_6.addWidget(self.process_image_widget)
        self.widget_4 = QtWidgets.QWidget(self.analysis_pic_plot_splitter)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_5.setContentsMargins(0, 3, 0, 0)
        self.verticalLayout_5.setSpacing(12)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_12 = QtWidgets.QLabel(self.widget_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_18.addWidget(self.label_12)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem7)
        self.p_enable_all_button = QtWidgets.QPushButton(self.widget_4)
        self.p_enable_all_button.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.p_enable_all_button.setFont(font)
        self.p_enable_all_button.setObjectName("p_enable_all_button")
        self.horizontalLayout_18.addWidget(self.p_enable_all_button)
        self.verticalLayout_5.addLayout(self.horizontalLayout_18)
        self.analysis_plots_splitter = QtWidgets.QSplitter(self.widget_4)
        self.analysis_plots_splitter.setStyleSheet("QSplitter::handle {\n"
"    image:url(:/icons/vert_splitter_handle.png)\n"
"}")
        self.analysis_plots_splitter.setOrientation(QtCore.Qt.Vertical)
        self.analysis_plots_splitter.setObjectName("analysis_plots_splitter")
        self.fit_widget = PlotWidget(self.analysis_plots_splitter)
        self.fit_widget.setObjectName("fit_widget")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.fit_widget)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.charge_widget = PlotWidget(self.analysis_plots_splitter)
        self.charge_widget.setObjectName("charge_widget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.charge_widget)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_5.addWidget(self.analysis_plots_splitter)
        self.verticalLayout_4.addWidget(self.analysis_pic_plot_splitter)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.horizontalLayout_16.addLayout(self.verticalLayout_2)
        self.line_7 = QtWidgets.QFrame(QuadScanDialog)
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.horizontalLayout_16.addWidget(self.line_7)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setContentsMargins(-1, -1, 6, -1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.eps_label = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.eps_label.setFont(font)
        self.eps_label.setObjectName("eps_label")
        self.gridLayout_4.addWidget(self.eps_label, 4, 2, 1, 1)
        self.label_37 = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.gridLayout_4.addWidget(self.label_37, 3, 0, 1, 1)
        self.result_label = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.result_label.setFont(font)
        self.result_label.setObjectName("result_label")
        self.gridLayout_4.addWidget(self.result_label, 0, 0, 1, 1)
        self.label_39 = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_39.setFont(font)
        self.label_39.setObjectName("label_39")
        self.gridLayout_4.addWidget(self.label_39, 8, 0, 1, 1)
        self.alpha_label = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.alpha_label.setFont(font)
        self.alpha_label.setObjectName("alpha_label")
        self.gridLayout_4.addWidget(self.alpha_label, 2, 2, 1, 1)
        self.label_35 = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.gridLayout_4.addWidget(self.label_35, 4, 0, 1, 1)
        self.status_textedit = QtWidgets.QTextEdit(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.status_textedit.sizePolicy().hasHeightForWidth())
        self.status_textedit.setSizePolicy(sizePolicy)
        self.status_textedit.setMinimumSize(QtCore.QSize(150, 0))
        self.status_textedit.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.status_textedit.setFont(font)
        self.status_textedit.setReadOnly(True)
        self.status_textedit.setObjectName("status_textedit")
        self.gridLayout_4.addWidget(self.status_textedit, 7, 2, 1, 1)
        self.label_38 = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_38.setFont(font)
        self.label_38.setObjectName("label_38")
        self.gridLayout_4.addWidget(self.label_38, 2, 0, 1, 1)
        self.label_40 = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_40.setFont(font)
        self.label_40.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_40.setObjectName("label_40")
        self.gridLayout_4.addWidget(self.label_40, 7, 0, 1, 1)
        self.mouse_label = QtWidgets.QLabel(QuadScanDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mouse_label.sizePolicy().hasHeightForWidth())
        self.mouse_label.setSizePolicy(sizePolicy)
        self.mouse_label.setMinimumSize(QtCore.QSize(150, 0))
        self.mouse_label.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.mouse_label.setFont(font)
        self.mouse_label.setObjectName("mouse_label")
        self.gridLayout_4.addWidget(self.mouse_label, 8, 2, 1, 1)
        self.beta_label = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.beta_label.setFont(font)
        self.beta_label.setObjectName("beta_label")
        self.gridLayout_4.addWidget(self.beta_label, 3, 2, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem8, 5, 0, 1, 1)
        self.result_axis_label = QtWidgets.QLabel(QuadScanDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.result_axis_label.setFont(font)
        self.result_axis_label.setObjectName("result_axis_label")
        self.gridLayout_4.addWidget(self.result_axis_label, 1, 2, 1, 1)
        self.horizontalLayout_16.addLayout(self.gridLayout_4)
        self.verticalLayout.addLayout(self.horizontalLayout_16)

        self.retranslateUi(QuadScanDialog)
        QtCore.QMetaObject.connectSlotsByName(QuadScanDialog)
        QuadScanDialog.setTabOrder(self.load_disk_button, self.p_raw_image_radio)
        QuadScanDialog.setTabOrder(self.p_raw_image_radio, self.p_filtered_image_radio)
        QuadScanDialog.setTabOrder(self.p_filtered_image_radio, self.p_roi_cent_x_spinbox)
        QuadScanDialog.setTabOrder(self.p_roi_cent_x_spinbox, self.p_roi_cent_y_spinbox)
        QuadScanDialog.setTabOrder(self.p_roi_cent_y_spinbox, self.p_roi_size_w_spinbox)
        QuadScanDialog.setTabOrder(self.p_roi_size_w_spinbox, self.p_roi_size_h_spinbox)
        QuadScanDialog.setTabOrder(self.p_roi_size_h_spinbox, self.p_threshold_spinbox)
        QuadScanDialog.setTabOrder(self.p_threshold_spinbox, self.p_keep_charge_ratio_spinbox)
        QuadScanDialog.setTabOrder(self.p_keep_charge_ratio_spinbox, self.p_median_kernel_spinbox)
        QuadScanDialog.setTabOrder(self.p_median_kernel_spinbox, self.fit_algo_combobox)
        QuadScanDialog.setTabOrder(self.fit_algo_combobox, self.p_x_radio)
        QuadScanDialog.setTabOrder(self.p_x_radio, self.p_y_radio)
        QuadScanDialog.setTabOrder(self.p_y_radio, self.p_image_index_slider)
        QuadScanDialog.setTabOrder(self.p_image_index_slider, self.p_enable_all_button)
        QuadScanDialog.setTabOrder(self.p_enable_all_button, self.status_textedit)

    def retranslateUi(self, QuadScanDialog):
        _translate = QtCore.QCoreApplication.translate
        QuadScanDialog.setWindowTitle(_translate("QuadScanDialog", "Dialog"))
        self.load_disk_button.setText(_translate("QuadScanDialog", "LOAD"))
        self.data_source_label.setText(_translate("QuadScanDialog", "--- No data loaded ---"))
        self.label_33.setText(_translate("QuadScanDialog", "Axis"))
        self.label_30.setText(_translate("QuadScanDialog", "Fit algorithm"))
        self.label_27.setText(_translate("QuadScanDialog", "ROI size (w / h)"))
        self.p_quad_screen_dist_label.setText(_translate("QuadScanDialog", "-.-- m"))
        self.label_24.setText(_translate("QuadScanDialog", "Quad screen dist"))
        self.p_electron_energy_label.setText(_translate("QuadScanDialog", "--- MeV"))
        self.label_29.setText(_translate("QuadScanDialog", "Median kernel"))
        self.p_quad_length_label.setText(_translate("QuadScanDialog", "-.-- m"))
        self.label_28.setText(_translate("QuadScanDialog", "Background level"))
        self.label_26.setText(_translate("QuadScanDialog", "ROI cent (x, y)"))
        self.label_25.setText(_translate("QuadScanDialog", "Electron energy"))
        self.p_x_radio.setText(_translate("QuadScanDialog", "x"))
        self.p_y_radio.setText(_translate("QuadScanDialog", "y"))
        self.label_32.setText(_translate("QuadScanDialog", "Select image"))
        self.label_23.setText(_translate("QuadScanDialog", "Quad length"))
        self.p_raw_image_radio.setText(_translate("QuadScanDialog", "Raw"))
        self.p_filtered_image_radio.setText(_translate("QuadScanDialog", "Processed"))
        self.label_11.setText(_translate("QuadScanDialog", "Show"))
        self.p_keep_charge_ratio_spinbox.setSuffix(_translate("QuadScanDialog", " %"))
        self.label_31.setText(_translate("QuadScanDialog", "Fit using charge"))
        self.process_button.setText(_translate("QuadScanDialog", "Process Now"))
        self.init_button.setText(_translate("QuadScanDialog", "Init"))
        self.p_k_value_label.setText(_translate("QuadScanDialog", "k = -.-- 1/m²"))
        self.p_k_ind_label.setText(_translate("QuadScanDialog", "k index 0/0"))
        self.p_image_label.setText(_translate("QuadScanDialog", "image 0/0"))
        self.label_12.setText(_translate("QuadScanDialog", "Fit"))
        self.p_enable_all_button.setText(_translate("QuadScanDialog", "Enable All"))
        self.eps_label.setText(_translate("QuadScanDialog", "--.- mm x mmrad"))
        self.label_37.setText(_translate("QuadScanDialog", "β"))
        self.result_label.setText(_translate("QuadScanDialog", "Result"))
        self.label_39.setText(_translate("QuadScanDialog", "Mouse"))
        self.alpha_label.setText(_translate("QuadScanDialog", "--.-"))
        self.label_35.setText(_translate("QuadScanDialog", "ε<sub>N</sub>"))
        self.label_38.setText(_translate("QuadScanDialog", "α"))
        self.label_40.setText(_translate("QuadScanDialog", "Status"))
        self.mouse_label.setText(_translate("QuadScanDialog", "--"))
        self.beta_label.setText(_translate("QuadScanDialog", "--.- m"))
        self.result_axis_label.setText(_translate("QuadScanDialog", "x-axis"))
from pyqtgraph import ImageView, PlotWidget
import quadscan_res_rc
