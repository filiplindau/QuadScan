# -*- coding:utf-8 -*-
"""
Created on Dec 18, 2017

@author: Filip
"""
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
import PyTango as pt
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
import numpy as np
import time
import sys
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class QTangoColors(object):
    def __init__(self):
        self.backgroundColor = '#000000'
        self.primaryColor0 = '#ff9900'
        self.primaryColor1 = '#ffcc66'
        self.primaryColor2 = '#feff99'
        self.primaryColor3 = '#bb6622'
        self.primaryColor4 = '#aa5533'
        self.primaryColor5 = '#882211'
        self.secondaryColor0 = '#66cbff'
        self.secondaryColor1 = '#3399ff'
        self.secondaryColor2 = '#99cdff'
        self.secondaryColor3 = '#3366cc'
        self.secondaryColor4 = '#000088'
        self.tertiaryColor0 = '#cc99cc'
        self.tertiaryColor1 = '#cc6699'
        self.tertiaryColor2 = '#cc6666'
        self.tertiaryColor3 = '#664466'
        self.tertiaryColor4 = '#9977aa'

        self.faultColor = '#ff0000'
        self.alarmColor2 = '#ffffff'
        self.warnColor = '#a35918'
        self.alarmColor = '#ff0000'
        self.warnColor2 = '#ffcc33'
        self.onColor = '#99dd66'
        self.offColor = '#ffffff'
        self.standbyColor = '#9c9cff'
        self.unknownColor = '#45616f'
        self.disableColor = '#ff00ff'
        self.movingColor = '#feff99'
        self.runningColor = '#feff99'

        self.validColor = self.secondaryColor0
        self.invalidColor = self.unknownColor
        self.changingColor = self.secondaryColor1

        self.legend_color_list = [self.secondaryColor0,
                                  self.primaryColor2,
                                  self.tertiaryColor1,
                                  self.secondaryColor3,
                                  self.primaryColor3,
                                  self.tertiaryColor4,
                                  self.primaryColor5,
                                  self.secondaryColor4,
                                  self.tertiaryColor3]


class QTangoSizes(object):
    def __init__(self):
        self.barHeight = 30
        self.barWidth = 20
        self.readAttributeWidth = 200
        self.readAttributeHeight = 200
        self.writeAttributeWidth = 280
        self.trendWidth = 100
        self.fontSize = 12
        self.fontType = 'Calibri'
        self.fontStretch = 75
        self.fontWeight = 50


def to_precision2(x, p=-1, w=-1, neg_compensation=False, return_prefix_string=True):
    """
    returns a string representation of x formatted with a precision of p OR width w

    """
    prefix_dict = {12: "T", 9: "G", 6: "M", 3: "k", 0: "", -3: "m", -6: "\u00b5", -9: "n", -12: "p", -15: "f", -18: "a"}
    out = []
    x = float(x)
    if np.isnan(x):
        return " " * (w-2) + "--"

    if x < 0:
        s_neg = "-"
        x = -x
    else:
        if neg_compensation:
            s_neg = " "
        else:
            s_neg = ""

    if x == 0.0:
        prefix = 0
    elif x < 1.0:
        prefix = int((np.log10(x) - 0.5) // 3)
    else:
        prefix = int((np.log10(x)) // 3)

    if p > 0:
        if p > w - 3:
            p = w - 3
    elif w > 3:
        p = w - 4
    else:
        p = 0
        w = 4

    s_val = "{0:.{p}f} ".format(x * 10**(-prefix*3), p=p)
    out.append(s_val)
    if return_prefix_string:
        out.append(prefix_dict[prefix*3])
        # if prefix != 0:
        #     w -= 1
        s_prefix = prefix_dict[prefix*3]
    else:
        s_prefix = ""

    s_len = len(s_val) + len(s_neg) + len(s_prefix)
    n_space = w - s_len
    if n_space < -1:        # Allow one extra since . is small
        s_val = s_val[:n_space+1]
        s = s_neg + s_val + s_prefix
    else:
        s = " " * n_space + s_neg + s_val + s_prefix

    # s = "".join(out)
    # n_space = w - len(s)
    # s = " " * n_space + s

    return s


class QTangoStripTool(QtWidgets.QFrame):
    def __init__(self, name=None, legend_pos="top", sizes=None, colors=None, chronological=True, parent=None):
        super().__init__(parent)
        logger.info("CREATING STRIPTOOL")
        self.setObjectName(self.__class__.__name__)
        self.colors = QTangoColors()    # type: QTangoColors
        self.sizes = QTangoSizes()
        self.attr_colors = self.colors

        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(QtWidgets.QHBoxLayout())
        self.inner_layout = QtWidgets.QVBoxLayout()

        self.legend_widget = QTangoStripToolLegendWidget(legend_pos, sizes, colors)
        self.plot_widget = QTangoStripToolPlotWidget(None, sizes, colors, chronological)
        self.name = None
        if isinstance(name, str):
            self.name = name
            self.add_curve(name)
            self.legend_widget.items.get(name).clicked.emit()
        logger.debug("Plot widget created")
        logger.debug("Legend widget created")

        self.set_legend_position(legend_pos)

        s = str(self.styleSheet())
        use_background_color = True
        bkg_color = "#000000"
        color = "#a07e30"
        st = """QTangoStripTool {{
                      border-width: 0px;
                      border-color: {0};
                      border-style: solid;
                      border-radius: 0px;
                      margin: 1px;
                      color: {0};
                      background-color: {1};
                      }}""".format(color, bkg_color)
        self.setStyleSheet(st)
        self.plot_widget.update_curve_range_signal.connect(self.set_range)

    def add_curve(self, name, curve=None, unit=None, visible=True):
        curve_new = self.plot_widget.addCurve(name, curve)
        curve_new.sigClicked.connect(self.set_curve_focus)
        plot_color = curve_new.opts["pen"].color().name()
        logger.debug("Adding curve {0} with color {1}".format(name, plot_color))
        legend_item = QTangoStripToolLegendItem(name, unit=unit, color=plot_color, sizes=self.sizes, colors=self.colors)
        legend_item.clicked.connect(self.set_curve_focus)
        legend_item.show_check.toggled.connect(self.toggle_curve_show)
        self.legend_widget.addItem(legend_item)
        self.set_curve_visible(name, visible)

    def remove_curve(self, name):
        logger.info("Removing curve {0}".format(name))
        self.plot_widget.removeCurve(name)
        self.legend_widget.removeItem(name)

    def set_curve_visible(self, name, visible):
        self.legend_widget.items[name].show_check.setChecked(visible)

    def add_point(self, data, curve_index=0, auto_range=True):
        self.plot_widget.addPoint(data, curve_index, auto_range)
        if auto_range:
            legend_item = self.legend_widget.get_item(curve_index)
            axis_range = self.plot_widget.get_curve_range(curve_index)
            legend_item.set_range(axis_range[1])
        # logger.info("Striptool add point: {0:.1f} ms, autorange: {1:.1f} ms".format((t1-t0)*1e3, (t2-t1)*1e3))

    def set_data(self, x_data, y_data, curve_index=0, **kargs):

        self.plot_widget.setData(x_data, y_data, curve_index=curve_index, **kargs)
        if "auto_range" in kargs:
            auto_range = kargs["auto_range"]
        else:
            auto_range = True
        if auto_range:
            legend_item = self.legend_widget.get_item(curve_index)
            axis_range = self.plot_widget.get_curve_range(curve_index)
            legend_item.set_range(axis_range[1])

    def set_curve_focus(self):
        s = self.sender()
        if isinstance(s, QTangoStripToolLegendItem):
            logger.debug("{0}: Signal from {1}".format(self.__class__, s.name))
            ind = self.legend_widget.get_item_index(s)
            logger.debug("Legend item index {0}".format(ind))
            self.plot_widget.setCurveFocus(ind)
            self.legend_widget.set_focus_item(ind)
        else:
            ind = self.plot_widget.get_curve_focus_ind()
            self.plot_widget.get_curve_color(ind)
            self.legend_widget.set_focus_item(ind)

    def toggle_curve_show(self, state):
        name = self.sender().parent().name
        logger.info("Checkbox {0} new state {1}".format(name, state))
        if state:
            self.plot_widget.set_curve_visible(name, True)
        else:
            self.plot_widget.set_curve_visible(name, False)

    def set_legend_position(self, position):
        try:
            for w in range(2):
                self.inner_widget.layout().takeAt(0)
        except AttributeError:
            pass
        self.legend_widget.set_position(position)
        if position == "top":
            lay = QtWidgets.QVBoxLayout()
            lay.addWidget(self.legend_widget)
            lay.addWidget(self.plot_widget)
        elif position == "bottom":
            lay = QtWidgets.QVBoxLayout()
            lay.addWidget(self.plot_widget)
            lay.addWidget(self.legend_widget)
        elif position == "right":
            lay = QtWidgets.QHBoxLayout()
            lay.addWidget(self.plot_widget)
            lay.addWidget(self.legend_widget)
            w = self.width()
            self.legend_widget.setMaximumWidth(200)
            self.legend_widget.setMinimumWidth(200)
        else:
            lay = QtWidgets.QHBoxLayout()
            lay.addWidget(self.legend_widget)
            lay.addWidget(self.plot_widget)
            self.legend_widget.setMaximumWidth(200)
            self.legend_widget.setMinimumWidth(200)

        lay.setContentsMargins(0, 0, 0, 0)
        self.layout().takeAt(0)
        self.layout().addLayout(lay)
        # old_lay = self.layout()
        # if old_lay is not None:
        #     old_lay.deleteLater()
        self.inner_layout = lay
        # self.legend_widget.set_position(position)

    def set_range(self, name, r_min, r_max):
        # logger.info("Setting range for curve {0}: {1:.2f} - {2:.2f}".format(name, r_min, r_max))
        self.legend_widget.items[name].set_range([r_min, r_max])

    def paintEvent(self, a0):
        super(QTangoStripTool, self).paintEvent(a0)
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        s = self.style()
        s.drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)


class DummyLabel(QtWidgets.QFrame):
    def __init__(self):
        QtWidgets.QFrame.__init__(self)
        self.setObjectName(self.__class__.__name__)
        self.l1 = QtWidgets.QLabel("test")
        self.l2 = QtWidgets.QLabel("apa")
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.l1)
        self.layout().addWidget(self.l2)
        self.attrColors = QTangoColors()
        st = ''.join(("DummyLabel {\n",
                      'border-width: 2px; \n',
                      'border-color: ', self.attrColors.secondaryColor0, '; \n',
                      'border-style: solid; \n',
                      'border-radius: 0px; \n',
                      'padding: 2px; \n',
                      'margin: 1px; \n',
                      'color: ', self.attrColors.secondaryColor0, "; \n",
                      'background-color: ', self.attrColors.backgroundColor, ';}\n',
                      "QLabel {\n",
                      'border-width: 0px; \n',
                      'border-color: ', self.attrColors.secondaryColor0, '; \n',
                      'border-style: solid; \n',
                      'border-radius: 0px; \n',
                      'padding: 2px; \n',
                      'margin: 1px; \n',
                      'color: ', self.attrColors.secondaryColor0, "; \n",
                      'background-color: ', self.attrColors.backgroundColor, ';}'
                      ))
        self.setStyleSheet(st)


class QTangoStripToolLegendItem(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal()

    def __init__(self, name, range=[0, 1], color=None, unit=None, sizes=None, colors=None, parent=None):
        QtWidgets.QFrame.__init__(self)
        self.setObjectName(self.__class__.__name__)
        self.attrColors = QTangoColors()
        self.sizes = QTangoSizes()
        logger.debug("New legend item: {0}, {1}".format(name, self.objectName()))
        self.name = name
        self.range = range
        self.unit = unit
        if color is None:
            self.color = self.attrColors.secondaryColor0
        else:
            self.color = color
        self.border_width = 1
        lay = QtWidgets.QHBoxLayout()
        self.name_label = QtWidgets.QLabel(name)
        self.range_label = QtWidgets.QLabel("[{0}-{1}] ".format(to_precision2(range[0], 2, 4, True),
                                                                to_precision2(range[1], 2, 4, True)))
        self.unit_label = QtWidgets.QLabel(unit)
        self.show_check = QtWidgets.QRadioButton(parent=self)
        self.show_check.setChecked(True)
        lay.addWidget(self.name_label)
        lay.addSpacerItem(QtWidgets.QSpacerItem(3, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
        lay.addWidget(self.range_label)
        lay.addWidget(self.unit_label)
        lay.addWidget(self.show_check)
        self.setLayout(lay)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.update_stylesheet()

    def update_stylesheet(self, new_color=None, new_width=None):
        if new_color is not None:
            self.color = new_color
        if new_width is not None:
            self.border_width = new_width
        st = """QTangoStripToolLegendItem {{
                      border-top-width: {2}px;
                      border-bottom-width: {2}px;
                      border-left-width: {3}px;
                      border-right-width: {3}px;
                      border-color: {0};
                      border-style: solid;
                      border-radius: 0px;
                      padding-left: {4}px;
                      padding-right: {4}px;
                      padding-top: 1px;
                      padding-bottom: 1px;
                      color: {0};
                      background-color: {1};
                      }}
                QLabel {{
                      border-width: 0px;
                      border-color: {0};
                      border-style: solid;
                      border-radius: 0px;
                      padding: 0px;
                      margin: 0px;
                      color: {0};
                      background-color: {1};
                      }}
                QLabel:hover {{
                      border-color: {0};
                      }}
                QRadioButton::indicator{{
                    width: 8px;
                    height: 8px;
                    border-width: 2px;
                    border-color: {0};
                    border-style: solid;
                    border-radius: 0px;
                    margin: 1px;
                    color: #008833;
                    background-color: {1};
                    }}

                QRadioButton::indicator::checked{{
                    border-color: {0};
                    background-color: {0};
                }}
                      """.format(self.color, self.attrColors.backgroundColor, 1, self.border_width*4, 8 - self.border_width*4)
        self.setStyleSheet(st)
        self.update()

    def set_name(self, name):
        self.name = name
        self.name_label.setText(name)

    def set_range(self, range):
        self.range = range
        self.range_label.setText("[{0}-{1}] ".format(to_precision2(range[0], 2, 4, True),
                                                     to_precision2(range[1], 2, 4, True)))

    def set_unit(self, unit):
        self.unit = unit
        self.unit_label.setText(unit)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == QtCore.Qt.LeftButton:
            logger.debug("{0}: Emitting clicked signal".format(self.name))
            self.clicked.emit()


class QTangoStripToolLegendWidget(QtWidgets.QWidget):
    def __init__(self, position="bottom", sizes=None, colors=None, parent=None):
        super().__init__(parent)
        self.items = OrderedDict()
        self.item_name_list = list()
        self.legend_gridlayout = QtWidgets.QGridLayout()
        self.inner_layout = QtWidgets.QHBoxLayout()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addLayout(self.inner_layout)
        self.max_col = None
        self.position = position
        self.set_position(position)
        self.current_focus_item = None

        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, legend_item):
        self.items[legend_item.name] = legend_item
        self.item_name_list.append(legend_item.name)
        self.set_position(self.position)

    def removeItem(self, item):
        logger.debug("Removing item {0}".format(item))
        if isinstance(item, QTangoStripToolLegendItem):
            it = self.items.pop(item.name)
        else:
            it = self.items.pop(item)
        self.item_name_list.remove(it.name)
        it.deleteLater()
        self.set_position(self.position)

    def get_item(self, item_id) -> QTangoStripToolLegendItem:
        if isinstance(item_id, str):
            i = self.items[item_id]
        else:
            # Assume item is an index
            i = self.items[self.item_name_list[item_id]]
        return i

    def get_item_index(self, item):
        try:
            ind = self.item_name_list.index(item.name)
        except ValueError:
            ind = None
        return ind

    def set_focus_item(self, item):
        if isinstance(item, str):
            i = self.items[item]
        else:
            # Assume item is an index
            i = self.items[self.item_name_list[item]]
        if self.current_focus_item is not None:
            self.current_focus_item.update_stylesheet(new_width=1)
        i.update_stylesheet(new_width=3)
        self.current_focus_item = i
        logger.debug("Setting focus stylesheet to {0}".format(i.name))

    def del_lay(self):
        for i in range(len(self.items)):
            self.legend_gridlayout.takeAt(0)
        self.inner_layout.takeAt(0)
        self.layout().takeAt(0)

    def set_position(self, position):
        self.del_lay()
        self.position = position

        if position in ["bottom", "top"]:
            self.max_col = 4
            lay = QtWidgets.QHBoxLayout()
            spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            lay.addSpacerItem(spacer)
        else:
            self.max_col = 1
            lay = QtWidgets.QVBoxLayout()
            spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            lay.addSpacerItem(spacer)
        n = len(self.items) - 1
        cm = n % self.max_col + 1
        rm = np.maximum(n // self.max_col, 1)
        grid_lay = QtWidgets.QGridLayout()
        c = 0
        r = 0

        for item in self.items.values():
            logger.debug("Position {4}, Adding legend item at {0}, {1}, rm {2}, cm {3}, max_col {5}".format(r, c, rm, cm, position, self.max_col))
            grid_lay.addWidget(item, r, c)
            c += 1
            if c % cm == 0:
                c = 0
                r += 1
        lay.addLayout(grid_lay)
        self.inner_layout = lay
        self.layout().addLayout(lay)


class QTangoStripToolPlotWidget(pg.PlotWidget):
    """ Base class for a trend widget.

        The widget stores trend curves that are trended. The duration is set with setDuration (seconds).
        Curves are added with addCurve. New points are added with addPoint.

        If curves are named with setCurveName they can be shown with showLegend.

    """
    update_curve_range_signal = QtCore.pyqtSignal(str, float, float)

    def __init__(self, name=None, sizes=None, colors=None, chronological=True, parent=None):
        pg.PlotWidget.__init__(self, useOpenGL=True)

        self.unselected_pen_width = 1.5
        self.selected_pen_width = 3.0
        self.unselected_pen_alpha = 0.5
        self.selected_pen_alpha = 0.8

        self.attrColors = QTangoColors()
        self.sizes = QTangoSizes()

        self.values_size = 10000
        self.duration = 600.0
        self.x_values = list()
        self.y_values = list()

        self.legend = None
        self.curve_focus = 0
        self.curve_name_list = list()
        self.curve_vb_list = list()
        self.curve_ax_list = list()
        self.value_trend_curves = list()
        self.current_data_index = list()

        self.trend_menu = None

        self.chronological = chronological

        self.setupLayout(name)
        self.setupTrendMenu()
        if name is not None:
            self.addCurve(name)
            self.setCurveFocus(0)
        # self.setupData()

    def setupLayout(self, name=None):
        self.setXRange(-self.duration, 0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        pi = self.getPlotItem()
        ax_left = pi.getAxis('left')
        ax_left.setPen(self.attrColors.secondaryColor0)
        pi.hideAxis('left')
        ax_bottom = pi.getAxis('bottom')
        ax_bottom.setPen(self.attrColors.secondaryColor0)

        ax_right = pi.getAxis('right')
        ax_right.setPen(self.attrColors.secondaryColor0)
        # 		ax_right.setWidth(0)
        # 		ax_right.setTicks([])
        # 		ax_right.showLabel(False)
        pi.showAxis('right')
        pi.sigYRangeChanged.connect(self.updateViews)
        # pi.sigXRangeChanged.connect(self.updateViews)
        pi.vb.sigResized.connect(self.updateViews)
        # pi.vb.sigRangeChanged.connect(self.updateViews)
        pi.autoBtn.clicked.connect(self.auto_range_all)

        color_warn = QtGui.QColor(self.attrColors.warnColor)
        color_warn.setAlphaF(0.75)
        color_good = QtGui.QColor(self.attrColors.secondaryColor0)
        color_good.setAlphaF(0.33)

        # self.legend = self.getPlotItem().addLegend()

    def setupData(self, curve=0):
        """ Pre-allocate data arrays
        """
        try:
            # Curve already exists
            self.x_values[curve] = -np.ones(self.values_size) * np.inf
            self.y_values[curve] = np.zeros(self.values_size)
            self.current_data_index[curve] = 0
            logger.debug("Setting up data for curve {0}".format(curve))
            self.value_trend_curves[curve].setData(self.x_values[curve], self.y_values[curve], antialias=True)
        except IndexError:
            # Need to create new arrays
            logger.debug("Adding new data arrays for curve {0}".format(curve))
            self.x_values.append(-np.ones(self.values_size) * np.inf)
            self.y_values.append(np.zeros(self.values_size))
            self.current_data_index.append(0)
            # if len(self.value_trend_curves) < curve + 1:
            #     self.addCurve()
            # self.valueTrendCurves[curve].setData(self.xValues[curve], self.yValues[curve], antialias = True)

    def setupTrendMenu(self):
        pi = self.getPlotItem()
        self.trend_menu = QtWidgets.QMenu()
        self.trend_menu.setTitle("Trend options")
        duration_action = QtWidgets.QWidgetAction(self)
        duration_widget = QtWidgets.QWidget()
        duration_layout = QtWidgets.QHBoxLayout()
        duration_label = QtWidgets.QLabel("Duration / s")
        duration_spinbox = QtWidgets.QDoubleSpinBox()
        duration_spinbox.setMaximum(3e7)
        duration_spinbox.setValue(self.duration)
        duration_spinbox.setMinimumWidth(40)
        duration_spinbox.editingFinished.connect(self.setDurationContext)

        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(duration_spinbox)
        duration_widget.setLayout(duration_layout)
        duration_action.setDefaultWidget(duration_widget)
        self.trend_menu.addAction(duration_action)
        pi.ctrlMenu = [self.trend_menu, pi.ctrlMenu]

    def setWarningLimits(self, limits):
        if type(limits) == pt.AttributeInfoListEx:
            warn_high = limits[0].alarms.max_warning
            warn_low = limits[0].alarms.min_warning
        else:
            warn_low = limits[0]
            warn_high = limits[1]
        self.warningRegionUpper.setRegion([warn_high, 1e6])
        self.warningRegionLower.setRegion([-1e6, warn_low])
        self.goodRegion.setRegion([warn_low, warn_high])

    def configureAttribute(self, attr_info):
        # QTangoAttributeBase.configureAttribute(self, attr_info)
        try:
            min_warning = float(self.attrInfo.alarms.min_warning)
        except ValueError:
            min_warning = -np.inf
        try:
            max_warning = float(self.attrInfo.alarms.max_warning)
        except ValueError:
            max_warning = np.inf
        self.setWarningLimits((min_warning, max_warning))
        self.setUnit(self.attrInfo.unit)

    def setDuration(self, duration):
        """ Set the duration of the trend graph in x axis units
        (e.g. samples, seconds...)
        """
        self.duration = duration
        self.setXRange(-self.duration, 0)

    def setDurationContext(self):
        """ Set the duration of the trend graph in x axis units
        (e.g. samples, seconds...) from the context menu
        """
        w = self.sender()
        duration = w.value()
        self.duration = duration
        self.setXRange(-self.duration, 0)

    def addCurve(self, name=None, curve=None):
        curve_index = len(self.value_trend_curves)
        if name is None:
            name = str(curve_index)
        logger.info("Adding curve {0}, name {1}".format(curve_index, name))
        vb = pg.ViewBox()
        vb.setZValue(-100)
        ax = pg.AxisItem("right")
        ax.linkToView(vb)
        ax1 = pg.AxisItem("bottom")
        ax1.linkToView(vb)
        # vb.enableAutoRange("y")
        # vb.setRange(yRange=[-10, 10])
        pi_main = self.getPlotItem()
        pi_main.scene().addItem(vb)
        vb.setXLink(pi_main)
        # vb.setYLink(pi_main)

        if curve is None:
            curve_new = pg.PlotCurveItem(name=name, antialias=True)
            curve_color = pg.mkColor(self.attrColors.legend_color_list[curve_index % len(self.attrColors.legend_color_list)])
            curve_color.setAlphaF(self.unselected_pen_alpha)
            curve_new.setPen(curve_color, width=self.unselected_pen_width)
            curve_new.setClickable(True)
        else:
            curve_new = curve
            pen = curve.opts["pen"]
            brush = curve.opts["brush"]
            logger.debug("Provided curve pen: {0}, brush: {1}".format(pen.color().name(), brush.color().name()))
        curve_new.setZValue(-100)
        vb.addItem(curve_new)

        logger.debug("New curve color: {0}".format(curve_new.opts["pen"].color().name()))

        self.value_trend_curves.append(curve_new)
        self.curve_vb_list.append(vb)
        self.curve_ax_list.append(ax)
        self.curve_name_list.append(name)

        # self.legend.addItem(curve_new, name)

        self.setupData(curve_index)

        curve_new.sigClicked.connect(self.setCurveFocus)
        # self.setCurveFocus(name)

        logger.debug("Calling updateViews")
        self.updateViews()
        return curve_new

    def removeCurve(self, name):
        logger.info("Removing curve {0}".format(name))
        ind = self.curve_name_list.index(name)
        vb = self.curve_vb_list.pop(ind)
        ax = self.curve_ax_list.pop(ind)
        curve = self.value_trend_curves.pop(ind)
        self.curve_name_list.pop(ind)
        self.getPlotItem().removeItem(vb)
        curve.deleteLater()
        ax.deleteLater()
        vb.deleteLater()

    def updateViews(self, data=None, data2=None):
        t0 = time.time()
        pi = self.getPlotItem()
        # logger.info("Curve index {0} selected".format(self.curve_focus))
        for ind, vb in enumerate(self.curve_vb_list):
            vb.setGeometry(pi.vb.sceneBoundingRect())
            vb.linkedViewChanged(pi.vb, vb.XAxis)
            if ind == self.curve_focus:
                vb.linkedViewChanged(pi.vb, vb.YAxis)
                name = self.curve_name_list[ind]
                vr = vb.viewRange()
                self.update_curve_range_signal.emit(name, vr[1][0], vr[1][1])
        # dt = time.time() - t0
        # logger.info("Updating view. {0:.1f} ms".format(dt * 1e3))

    def setCurveFocus(self, curve_id):
        curve_old = self.value_trend_curves[self.curve_focus]
        curve_old_color = curve_old.opts["pen"].color()
        curve_old_color.setAlphaF(self.unselected_pen_alpha)
        curve_old.setZValue(-100)
        curve_old.setPen(curve_old_color, width=self.unselected_pen_width)
        vb_old = self.curve_vb_list[self.curve_focus]
        vb_old.setZValue(-100)
        if isinstance(curve_id, int):
            name = self.curve_name_list[curve_id]
            self.curve_focus = curve_id
        elif isinstance(curve_id, str):
            name = curve_id
            self.curve_focus = self.curve_name_list.index(name)
        else:
            self.curve_focus = self.value_trend_curves.index(curve_id)
            name = self.curve_name_list[self.curve_focus]
        logger.debug("Curve {0} selected, index {1}".format(name, self.curve_focus))
        pi = self.getPlotItem()
        axis_viewrange = self.curve_vb_list[self.curve_focus].viewRange()
        logger.debug("Setting view range {0}".format(axis_viewrange[1]))
        pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
        pi_ax = pi.getAxis("right")
        curve_selected = self.value_trend_curves[self.curve_focus]
        curve_color = curve_selected.opts["pen"].color()
        curve_color.setAlphaF(self.selected_pen_alpha)
        curve_selected.setPen(curve_color, width=self.selected_pen_width)
        # curve_selected.setZValue(0.0)
        # self.curve_vb_list[self.curve_focus].sigYRangeChanged.connect(self.updateViews)
        # self.curve_vb_list[self.curve_focus].setZValue(0.0)
        logger.debug("Axis color: {0}".format(curve_color.getRgb()))
        pi_ax.setPen(curve_color)
        pi.showGrid(True, True, 0.4)
        self.updateViews()

    def get_curve_range(self, curve):
        if isinstance(curve, int):
            name = self.curve_name_list[curve]
            curve_index = curve
        elif isinstance(curve, str):
            name = curve
            curve_index = self.curve_name_list.index(name)
        else:
            name = curve.opts.get('name', None)
            curve_index = self.curve_name_list.index(name)
        axis_viewrange = self.curve_vb_list[curve_index].viewRange()
        return axis_viewrange

    def get_curve_color(self, curve_id) -> QtGui.QColor:
        if isinstance(curve_id, int):
            name = self.curve_name_list[curve_id]
            curve_index = curve_id
        elif isinstance(curve_id, str):
            name = curve_id
            curve_index = self.curve_name_list.index(name)
        else:
            name = curve_id.opts.get('name', None)
            curve_index = self.curve_name_list.index(name)
        curve_color = self.value_trend_curves[curve_index].opts["pen"].color()
        logger.debug("Curve color: {0}".format(curve_color))
        return curve_color

    def get_curve_focus_ind(self):
        return self.curve_focus

    def get_curve(self, curve_id):
        if isinstance(curve_id, int):
            name = self.curve_name_list[curve_id]
            curve_index = curve_id
        elif isinstance(curve_id, str):
            name = curve_id
            curve_index = self.curve_name_list.index(name)
        else:
            name = curve_id.opts.get('name', None)
            curve_index = self.curve_name_list.index(name)
        curve = self.value_trend_curves[curve_index]
        return curve

    def showLegend(self, show_legend=True):
        if show_legend is True:
            if self.legend is None:
                self.legend = self.addLegend(offset=(5, 5))
                for it in self.value_trend_curves:
                    self.legend.addItem(it, it.opts.get('name', None))
        else:
            if self.legend is not None:
                self.legend.scene().removeItem(self.legend)
                self.legend = None

    def setCurveName(self, curve, name):
        self.value_trend_curves[curve].opts['name'] = name

    def set_curve_visible(self, name, visible):
        ind = self.curve_name_list.index(name)
        if visible:
            self.value_trend_curves[ind].show()
        else:
            self.value_trend_curves[ind].hide()

    def addPoint(self, data, curve_index=0, auto_range=True):
        t0 = time.time()
        if type(data) == pt.DeviceAttribute:
            x_new = data.time.totime()
            y_new = data.value
        else:
            x_new = data[0]
            y_new = data[1]
        # Check x_new against last x to see if it is increasing.
        # Sometimes there is a bug with wrong time values that are very much lower
        # than the old value (probably 0)
        current_data_index = self.current_data_index[curve_index]
        if current_data_index == 0:
            x_old = 0.0
        else:
            x_old = self.x_values[curve_index][current_data_index]
        if (self.chronological is False) or (x_new > x_old):
            # Rescaling if the number of samples is too high
            if current_data_index + 1 >= self.values_size:
                current_data_index = int(self.values_size * 0.75)
                self.x_values[curve_index][0:current_data_index] = self.x_values[curve_index][self.values_size -
                                                                                              current_data_index:
                                                                                              self.values_size]
                self.y_values[curve_index][0:current_data_index] = self.y_values[curve_index][self.values_size -
                                                                                              current_data_index:
                                                                                              self.values_size]
            elif current_data_index == 0:
                self.x_values[curve_index][0] = x_new
                self.y_values[curve_index][0] = y_new
            current_data_index += 1
            self.x_values[curve_index][current_data_index] = x_new
            start_index = np.argmax((self.x_values[curve_index] - x_new) > -self.duration)
            self.y_values[curve_index][self.current_data_index[curve_index]] = y_new
            self.value_trend_curves[curve_index].setData(self.x_values[curve_index][start_index:current_data_index] - x_new,
                                                         self.y_values[curve_index][start_index:current_data_index],
                                                         antialias=False)
            if auto_range:
                vb = self.curve_vb_list[curve_index]
                vb.enableAutoRange("y")
                vb.autoRange()
                if self.curve_focus == curve_index:
                    pi = self.getPlotItem()
                    axis_viewrange = self.curve_vb_list[curve_index].viewRange()
                    # logger.debug("Setting view range {0}".format(axis_viewrange))
                    pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
                    pi.vb.setRange(xRange=axis_viewrange[0])
            self.current_data_index[curve_index] = current_data_index
            t1 = time.time()
            # self.update()
            t2 = time.time()
            # logger.info("Add point timing: setup {0:.1f} ms, update {1:.1f} ms".format((t1-t0)*1e3, (t2-t1)*1e3))

    # def setData(self, x_data, y_data, curve_index=0, auto_range=True):
    def setData(self, x_data, y_data, curve_index=0, auto_range=True, **kargs):
        if isinstance(curve_index, str):
            curve_index = self.curve_name_list.index(curve_index)
        logger.debug("Setting data for curve {0}".format(curve_index))
        self.setupData(curve_index)
        n = x_data.shape[0]
        self.x_values[curve_index][-n:] = x_data
        self.y_values[curve_index][-n:] = y_data
        vb = self.curve_vb_list[curve_index]
        # vb.enableAutoRange("y")
        if auto_range:
            vb.enableAutoRange(pg.ViewBox.XYAxes, True)
        else:
            vb.enableAutoRange(pg.ViewBox.XYAxes, False)
        self.value_trend_curves[curve_index].setData(x_data, y_data, **kargs)
        if "auto_range" in kargs:
            auto_range = kargs["auto_range"]
        else:
            auto_range = False
        if auto_range:
            vb.autoRange()
        if self.curve_focus == curve_index:
            pi = self.getPlotItem()
            axis_viewrange = self.curve_vb_list[curve_index].viewRange()
            logger.debug("Setting view range {0}".format(axis_viewrange))
            pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
            pi.vb.setRange(xRange=axis_viewrange[0])
        self.updateViews()

    def autoScale(self, curve_ind=None):
        if curve_ind is not None:
            vb = self.curve_vb_list[curve_ind]
            child_range = vb.childrenBounds(frac=1.0)
            vb.setRange("yRange", child_range[1])
        else:
            for vb in self.curve_vb_list:
                child_range = vb.childrenBounds(frac=[1.0, 1.0])
                vb.setRange(yRange=child_range[1])

    def scaleAll(self, sc, center=None):
        for ind, vb in enumerate(self.curve_vb_list):
            if center is None:
                vr = vb.targetRect()
                center = vr.center().y()
            top = center + sc * (vr.top() - center)
            bottom = center + sc * (vr.bottom() - center)
            logger.info("Curve {0}: vr {1}, center {2}, top {3}, bottom {4}".format(ind, vr, center, top, bottom))
            self.setYRange(top, bottom, padding=0)

    def auto_range_all(self):
        logger.info("Auto ranging all curves.")
        pi = self.getPlotItem()
        x_min = np.inf
        x_max = -np.inf
        for ind, vb in enumerate(self.curve_vb_list):
            vb.autoRange(padding=0.05)
            vr = vb.viewRange()
            x_min = np.minimum(x_min, vr[0][0])
            x_max = np.maximum(x_max, vr[0][1])
            name = self.curve_name_list[ind]
            self.update_curve_range_signal.emit(name, vr[1][0], vr[1][1])
            # vb.setGeometry(pi.vb.sceneBoundingRect())
            # vb.linkedViewChanged(pi.vb, vb.XAxis)
            if ind == self.curve_focus:
                pi.vb.setRange(yRange=vr[1], padding=0.1)
        pi.vb.setRange(xRange=[x_min, x_max], padding=0.05)


class TestStream(QtWidgets.QWidget):
    update_sig = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.strip_tool = QTangoStripTool("Test", legend_pos="right")
        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setMinimumWidth(80)
        self.start_button = QtWidgets.QPushButton("START")
        self.start_button.clicked.connect(self.start2)
        self.start_button.setMinimumWidth(80)
        self.autoscale_button = QtWidgets.QPushButton("SCALE")
        self.autoscale_button.clicked.connect(self.autoscale)
        self.autoscale_button.setMinimumWidth(80)
        self.fps_label = QtWidgets.QLabel("FPS: -.-")
        l2 = QtWidgets.QHBoxLayout()
        l2.addWidget(self.start_button)
        l2.addWidget(self.stop_button)
        l2.addWidget(self.autoscale_button)
        l2.addWidget(self.fps_label)
        l2.addSpacerItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        l2.setSpacing(10)
        self.stop_thread_flag = False
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addLayout(l2)
        self.layout().addWidget(self.strip_tool)
        self.thread_list = list()
        self.n_points = 100
        self.n_curves = 4

        self.update_sig.connect(self.update_data)

        self.setup_curves()
        st = """QPushButton {{ border-width: 2px;
                      border-color: {1};
                      border-style: solid; 
                      border-radius: 0px;
                      color: {1};}}
                QLabel {{color: {1};}}
                QWidget {{
                              background-color: {0};
                              }}""".format(QTangoColors().backgroundColor, QTangoColors().secondaryColor0)
        self.setStyleSheet(st)
        self.start2()
        self.update()

    def test_data_stream(self, curve_index, update_time=0.5):
        logger.info("Thread {0} starting".format(curve_index))
        while not self.stop_thread_flag:
            x = time.time()
            y = np.random.random() * (curve_index + 1) + 2 * (curve_index + 1)
            self.strip_tool.add_point((x, y), curve_index, auto_range=False)
            t1 = time.time()
            time.sleep(update_time)
            t2 = time.time()
            logger.info("test data stream: add point {0:.1f} ms, total {1:.1f} ms".format((t1-x)*1e3, (t2-x)*1e3))
        logger.info("Thread {0} exiting".format(curve_index))

    def test_data_stream2(self):
        logger.info("Update thread starting")
        update_time = 0.03
        while not self.stop_thread_flag:
            x = np.zeros(self.n_curves)
            y = np.zeros_like(x)
            t0 = time.time()
            for curve_index in range(self.n_curves):
                y[curve_index] = np.random.random() * (curve_index + 1) + 2 * (curve_index + 1)
                x[curve_index] = t0
                # logger.info("c {0}".format(curve_index))
            t1 = time.time()
            self.update_sig.emit(x, y)
            time.sleep(update_time)
            t2 = time.time()
            # logger.info("test data stream: add point {0:.1f} ms, total {1:.1f} ms".format((t1-x)*1e3, (t2-x)*1e3))
            self.fps_label.setText("FPS: {0:.1f}".format(1.0 / (t2 - t0)))
        logger.info("Thread {0} exiting".format(curve_index))

    def update_data(self, x, y):
        for c in range(self.n_curves):
            self.strip_tool.add_point([x[c], y[c]], c, auto_range=False)

    def stop(self):
        logger.info("Stopping threads")
        self.stop_thread_flag = True
        for t in self.thread_list:
            t.join(1.0)
        # self.close()

    def start(self):
        self.stop()
        logger.info("Starting threads")
        self.stop_thread_flag = False
        for c in range(self.n_curves):
            t = threading.Thread(target=self.test_data_stream, args=(c, 0.1))
            t.start()
            self.thread_list.append(t)

    def start2(self):
        self.stop()
        logger.info("Starting update thread")
        self.stop_thread_flag = False
        t = threading.Thread(target=self.test_data_stream2)
        t.start()
        self.thread_list.append(t)

    def setup_curves(self):
        for c in range(self.n_curves-1):
            self.strip_tool.add_curve("Curve {0}".format(c + 1))
            # self.strip_tool.plot_widget.value_trend_curves[c].sigPlotChanged.connect(self.plot_changed)

        x0 = 0
        t0 = time.time()
        n_points = self.n_points * self.n_curves
        for p in range(self.n_points):
            x = x0 + p
            for c in range(self.n_curves):
                y = np.random.random() * (c + 1) + 2 * (c + 1)
                self.strip_tool.add_point((x, y), c)
        logger.info("Added {0:.1f} points/s".format(n_points / (time.time() - t0)))

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.stop()
        a0.accept()

    def plot_changed(self, c):
        logger.info("Curve {0} changed".format(c))

    def autoscale(self):
        # self.strip_tool.plot_widget.autoScale(None)
        self.strip_tool.plot_widget.scaleAll(1.2)


if __name__ == "__main__":
    pg.setConfigOptions(useOpenGL=True)

    app = QtWidgets.QApplication(sys.argv)

    test = "data"

    if test == "data":
        w = QtWidgets.QWidget()
        w.setLayout(QtWidgets.QHBoxLayout())
        strip_tool = QTangoStripTool(legend_pos="bottom")
        strip_tool2 = QTangoStripTool(legend_pos="right")
        # strip_tool.show()
        logger.debug("Strip tool created")
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(strip_tool)
        splitter.addWidget(strip_tool2)
        w.layout().addWidget(splitter)
        w.show()

        for c in range(3):
            x_data = np.linspace(-600, 0, 1000)
            y_data = np.sin(2*np.pi*x_data/240.0 * (c + 1)) + 10 * c
            strip_tool.add_curve("Curve {0}".format(c + 1))
            strip_tool.set_data(x_data, y_data, c)
            # strip_tool.curve_vb_list[c].setRange(yRange=[c-1, c+1])
        # strip_tool.set_legend_position("bottom")
        for c in range(5):
            x_data = np.linspace(-10, 10, 1000)
            y_data = x_data**(c % 3) + np.random.random(x_data.shape)
            strip_tool2.add_curve("Curve {0}".format(c + 1))
            strip_tool2.set_data(x_data, y_data, c)
        strip_tool2.remove_curve("Curve 2")

    elif test == "trend":
        test_stream = TestStream()
        test_stream.show()

    sys.exit(app.exec_())
