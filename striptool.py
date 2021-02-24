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
logger.setLevel(logging.INFO)


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
        # paletton.com:
        self.green1 = "#bbdfab"
        self.green2 = "#77c058"
        self.green3 = "#4aa224"
        self.green4 = "#2b8107"
        self.red1 = "#f1b9c3"
        self.red2 = "#d26075"
        self.red3 = "#b12840"
        self.red4 = "#8d081f"
        self.blue1 = "#9199b6"
        self.blue2 = "#4f6097"
        self.blue3 = "#283c7f"
        self.blue4 = "#112465"
        self.yellow1 = "#ffedc3"
        self.yellow2 = "#e0ba67"
        self.yellow3 = "#bc8f2a"
        self.yellow4 = "#966a08"

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

        self.legend_color_list = [self.green3, self.blue3, self.red3, self.yellow3,
                                  self.green2, self.blue2, self.red2, self.yellow2,
                                  self.green1, self.blue1, self.red1, self.yellow1,
                                  self.green4, self.blue4, self.red4, self.yellow4,
                                  self.secondaryColor0,
                                  self.primaryColor2,
                                  self.tertiaryColor1,
                                  self.secondaryColor3,
                                  self.primaryColor3,
                                  self.tertiaryColor4,
                                  self.primaryColor5,
                                  self.secondaryColor4,
                                  self.tertiaryColor3,
                                  ]


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
    points_clicked_signal = QtCore.pyqtSignal(str, list)

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

        self.legend_widget = QTangoStripToolLegendWidget(legend_pos, sizes, colors, parent=self)
        self.plot_widget = QTangoStripToolPlotWidget(None, sizes, colors, chronological, parent=self)
        self.plot_widget.points_clicked_signal.connect(self.points_clicked_signal)
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

    def add_curve(self, name, curve=None, unit=None, visible=True, **kwargs):
        legend_item = QTangoStripToolLegendItem(name, unit=unit, color=None, sizes=self.sizes, colors=self.colors)
        legend_item.clicked.connect(self.set_curve_focus)
        legend_item.show_check.toggled.connect(self.toggle_curve_show)
        self.legend_widget.addItem(legend_item)
        curve_new = self.plot_widget.addCurve(name, curve, **kwargs)
        curve_new.sigClicked.connect(self.set_curve_focus)
        if curve is None:
            curve_new.sigPointsClicked.connect(self.set_curve_focus)
        plot_color = self.plot_widget.get_curve_color(name).name()
        legend_item.update_stylesheet(plot_color)
        self.set_curve_visible(name, visible)
        self.set_y_link(name, name)
        logger.debug("Added curve {0} with color {1}".format(name, plot_color))

    def remove_curve(self, name):
        logger.info("Removing curve {0}".format(name))
        self.plot_widget.removeCurve(name)
        self.legend_widget.removeItem(name)

    def set_curve_visible(self, name, visible):
        self.legend_widget.items[name].show_check.setChecked(visible)

    def add_point(self, data, curve_name, auto_range=True):
        self.plot_widget.addPoint(data, curve_name, auto_range)
        if auto_range:
            legend_item = self.legend_widget.get_item(curve_name)
            axis_range = self.plot_widget.get_curve_range(curve_name)
            legend_item.set_range(axis_range[1])
        # logger.info("Striptool add point: {0:.1f} ms, autorange: {1:.1f} ms".format((t1-t0)*1e3, (t2-t1)*1e3))

    def set_data(self, x_data, y_data, curve_name, **kargs):

        self.plot_widget.setData(x_data, y_data, curve_name=curve_name, **kargs)
        if "auto_range" in kargs:
            auto_range = kargs["auto_range"]
        else:
            auto_range = True
        if auto_range:
            legend_item = self.legend_widget.get_item(curve_name)
            axis_range = self.plot_widget.get_curve_range(curve_name)
            legend_item.set_range(axis_range[1])

    def set_curve_focus(self):
        s = self.sender()
        if isinstance(s, QTangoStripToolLegendItem):
            logger.debug("{0}: Signal from {1}".format(self.__class__, s.name))
            ind = self.legend_widget.get_item_index(s)
            logger.debug("Legend item index {0}".format(ind))
            self.plot_widget.setCurveFocus(s.name)
            self.legend_widget.set_focus_item(s.name)
        else:
            ind = self.plot_widget.get_curve_focus_name()
            self.plot_widget.get_curve_color(ind)
            self.legend_widget.set_focus_item(ind)

    def toggle_curve_show(self, state):
        name = self.sender().parent().name
        logger.debug("Checkbox {0} new state {1}".format(name, state))
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
        logger.debug("Setting range for curve {0}: {1:.2f} - {2:.2f}".format(name, r_min, r_max))
        self.legend_widget.items[name].set_range([r_min, r_max])

    def set_y_link(self, curve_name_1, curve_name_2):
        logger.debug("c1: {0}, c2: {1}".format(curve_name_1, curve_name_2))
        old_group, new_group = self.plot_widget.set_y_link(curve_name_1, curve_name_2)
        logger.debug("Old group: {0}, new group {1}".format(old_group, new_group))
        color_list = list()
        for curve_name in old_group:
            color_list.append(self.plot_widget.get_curve_color(curve_name))
        for curve_name in old_group:
            li = self.legend_widget.get_item(curve_name)
            li.set_group_list(color_list)
        color_list = list()
        for curve_name in new_group:
            color_list.append(self.plot_widget.get_curve_color(curve_name))
        for curve_name in new_group:
            li = self.legend_widget.get_item(curve_name)
            li.set_group_list(color_list)

    def set_curve_color(self, curve_name, color):
        self.plot_widget.set_curve_color(curve_name, color)
        gr_list = self.plot_widget.get_link_group(curve_name)
        color_list = list()
        for curve_name in gr_list:
            color_list.append(self.plot_widget.get_curve_color(curve_name))
        for curve_name in gr_list:
            li = self.legend_widget.get_item(curve_name)
            li.set_group_list(color_list)

    def stack_vertically(self):
        self.plot_widget.stack_vertically()

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


class QTangoStripToolGroupWidget(QtWidgets.QWidget):
    def __init__(self, color_list=None, parent=None):
        super().__init__(parent)
        self.color_list = None
        self.set_color_list(color_list)
        logger.debug("Group widget created")
        self.setMinimumWidth(8)
        self.setMaximumWidth(8)

    def set_color_list(self, color_list):
        if not isinstance(color_list, list):
            color_list = list(color_list)
        self.color_list = list()
        for c in color_list:
            col = pg.mkColor(c)
            col.setAlphaF(1)
            self.color_list.append(col)
        self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        w = self.width()
        h = self.height()
        dy = h / len(self.color_list)
        painter.begin(self)
        brush = QtGui.QBrush(self.color_list[0])
        pen = QtGui.QPen(self.color_list[0])
        pen.setCapStyle(QtCore.Qt.FlatCap)
        pen.setWidth(w)
        for ind, c in enumerate(self.color_list):
            pen.setColor(c)
            painter.setPen(pen)
            brush.setColor(c)
            painter.setBrush(brush)
            painter.drawLine(QtCore.QPoint(int(w / 2), int(dy * ind)), QtCore.QPoint(int(w / 2), int(dy * (ind + 1))))
        painter.end()


class QTangoStripToolLegendItem(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal()

    def __init__(self, name, range=[0, 1], color=None, unit=None, sizes=None, colors=None, parent=None):
        QtWidgets.QFrame.__init__(self, parent=parent)
        two_row = True
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
        self.current_value_label = QtWidgets.QLabel()
        self.group_label = QTangoStripToolGroupWidget([self.color], parent=self)
        self.show_check = QtWidgets.QRadioButton(parent=self)
        self.show_check.setChecked(True)
        if two_row:
            lay_v = QtWidgets.QVBoxLayout()
            lay_h1 = QtWidgets.QHBoxLayout()
            lay_h2 = QtWidgets.QHBoxLayout()
            lay_h1.addWidget(self.name_label)
            lay_h1.addSpacerItem(QtWidgets.QSpacerItem(3, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
            lay_h1.addWidget(self.show_check)
            lay_h2.addWidget(self.range_label)
            lay_h2.addWidget(self.current_value_label)
            lay_h2.addSpacerItem(QtWidgets.QSpacerItem(3, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
            lay_h2.addWidget(self.unit_label)
            lay_v.addLayout(lay_h1)
            lay_v.addLayout(lay_h2)
            lay.addLayout(lay_v)
            lay.addWidget(self.group_label)
            lay.setContentsMargins(6, 2, 4, 2)
        else:
            lay.addWidget(self.name_label)
            lay.addSpacerItem(QtWidgets.QSpacerItem(3, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
            lay.addWidget(self.range_label)
            lay.addWidget(self.unit_label)
            lay.addWidget(self.show_check)
            lay.addWidget(self.group_label)
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
                      """.format(self.color, self.attrColors.backgroundColor, 1, self.border_width*3, 8 - self.border_width*3)
        logger.debug("Updating stylesheet for {0}:\n{1}".format(self.name, st))
        self.setStyleSheet(st)
        self.update()

    def set_name(self, name):
        self.name = name
        self.name_label.setText(name)

    def set_range(self, range):
        self.range = range
        self.range_label.setText("[{0}, {1}] ".format(to_precision2(range[0], 2, 4, True),
                                                     to_precision2(range[1], 2, 4, True)))

    def set_unit(self, unit):
        self.unit = unit
        self.unit_label.setText(unit)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == QtCore.Qt.LeftButton:
            logger.debug("{0}: Emitting clicked signal".format(self.name))
            self.clicked.emit()

    def set_group_list(self, color_group_list):
        self.group_label.set_color_list(color_group_list)

    def contextMenuEvent(self, a0: QtGui.QContextMenuEvent) -> None:
        context_menu = QtWidgets.QMenu(self)
        link_y_menu = context_menu.addMenu("Link y axis with")
        gr_list = self.parent().parent().plot_widget.curve_group_list
        gr_act_dict = dict()
        for gr in gr_list:
            act = QtWidgets.QAction("{0}".format(gr))
            link_y_menu.addAction(act)
            gr_act_dict[act] = gr
        link_y_menu.addSeparator()
        unlink_act = link_y_menu.addAction("Unlink")
        col_act = context_menu.addAction("Set color")
        as_act = context_menu.addAction("Autoscale this curve")
        as_act.setEnabled(False)
        vs_act = context_menu.addAction("Vertically stack curves")

        action = context_menu.exec_(self.mapToGlobal(a0.pos()))
        if action == col_act:
            # color: QtGui.QColor = QtWidgets.QColorDialog.getColor()
            col_diag = QtWidgets.QColorDialog()
            for ind, color in enumerate(self.attrColors.legend_color_list):
                col_diag.setCustomColor(ind, QtGui.QColor(color))
            result = col_diag.exec_()
            logger.debug("Colorpick returned {0}".format(result))
            color = col_diag.selectedColor()
            self.update_stylesheet(color.name())
            self.parent().parent().set_curve_color(self.name, color)
        elif action in gr_act_dict:
            self.parent().parent().set_y_link(self.name, gr_act_dict[action][0])
        elif action == unlink_act:
            self.parent().parent().set_y_link(self.name, self.name)
        elif action == vs_act:
            self.parent().parent().stack_vertically()


class QTangoStripToolLegendWidget(QtWidgets.QWidget):
    def __init__(self, position="bottom", sizes=None, colors=None, parent=None):
        super().__init__(parent=parent)
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
        legend_item.setParent(self)

    def removeItem(self, item):
        logger.debug("Removing item {0}".format(item))
        if isinstance(item, QTangoStripToolLegendItem):
            it = self.items.pop(item.name)
        else:
            it = self.items.pop(item)
        self.item_name_list.remove(it.name)
        self.set_position(self.position)        # The widget is removed from the layout here
        it.setParent(None)
        it.deleteLater()

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
    points_clicked_signal = QtCore.pyqtSignal(str, list)

    def __init__(self, name=None, sizes=None, colors=None, chronological=True, parent=None):
        pg.PlotWidget.__init__(self, useOpenGL=True)

        self.name = name

        self.unselected_pen_width = 1.5
        self.selected_pen_width = 3.0
        self.selected_pen_factor = 3
        self.unselected_pen_alpha = 0.5
        self.selected_pen_alpha = 0.8
        self.symbol_size = 10
        self.highlight_symbol_size = 20

        self.attrColors = QTangoColors()
        self.sizes = QTangoSizes()

        self.values_size = 10000
        self.duration = 600.0
        # self.x_values = list()
        # self.y_values = list()
        self.x_values_dict = dict()
        self.y_values_dict = dict()

        self.legend = None
        self.curve_focus = None                    # Index of currently selected curve (showing the y-axis)
        self.current_data_index = list()        # List of where the data is added for trend curves.
        self.next_curve_id_index = 0            # Next id when adding curve
        self.curve_group_list = list()          # Dict of curve groups. Each entry is a list of curve indices that are in the group
        self.curve_vb_dict = dict()             # Dict of viewboxes for curves. These set view area.
        self.curve_ax_dict = dict()             # Dict of axes for curves.
        self.curve_item_dict = dict()           # Dict of the actual curve objects (plotcurveitems or scatterplotitems)
        self.current_data_index_dict = dict()
        self.highlighted_point_dict = dict()

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

    def setupData(self, curve_name):
        """ Pre-allocate data arrays
        """
        self.x_values_dict[curve_name] = -np.ones(self.values_size) * np.inf
        self.y_values_dict[curve_name] = np.zeros(self.values_size)
        self.current_data_index_dict[curve_name] = 0
        logger.debug("Setting up data for curve {0}".format(curve_name))
        # self.curve_item_dict[curve_name].setData(self.x_values_dict[curve_name], self.y_values_dict[curve_name], antialias=True)

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

    def addCurve(self, name=None, curve=None, width=1, **kwargs):
        curve_index = self.next_curve_id_index
        self.next_curve_id_index += 1
        if name is None:
            name = str(curve_index)
        logger.info("Adding curve {0}, name {1}, width {2}".format(curve_index, name, width))
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
            curve_new = pg.PlotDataItem(name=name, antialias=True, **kwargs)
            if "color" in kwargs:
                curve_color = pg.mkColor(kwargs["color"])
            else:
                curve_color = pg.mkColor(
                    self.attrColors.legend_color_list[curve_index % len(self.attrColors.legend_color_list)])
            curve_color.setAlphaF(self.unselected_pen_alpha)
            if "pen" not in kwargs:
                if width is None:
                    pen = None
                    logger.info("Pen None")
                    # pen.setColor(curve_color)
                else:
                    pen = pg.mkPen(curve_color, width=width)
                curve_new.opts["pen"] = pen
            if "symbolBrush" not in kwargs:
                curve_new.setSymbolBrush(curve_color)
                curve_new.setSymbolPen(curve_color, width=0)
            if "symbolSize" not in kwargs:
                curve_new.setSymbolSize(self.symbol_size)
            # curve_new.setClickable(True)
            curve_new.curve.setClickable(True)
            curve_new.sigPointsClicked.connect(self.points_clicked)
        else:
            if isinstance(curve, pg.PlotDataItem):
                curve_new = curve
            elif isinstance(curve, pg.PlotCurveItem):
                curve_new = pg.PlotDataItem()
                curve_new.curve = curve

                curve_new.setPen(curve.opts["pen"])
                curve_new.setSymbolPen(curve.opts["pen"])
                curve_new.setSymbolBrush(curve.opts["brush"])
            elif isinstance(curve, pg.ScatterPlotItem):
                curve_new = pg.PlotDataItem()
                curve_new.scatter = curve
                curve_new.setPen(curve.opts["pen"])
                curve_new.setSymbolPen(curve.opts["pen"])
                curve_new.setSymbolBrush(curve.opts["brush"])
            pen = curve.opts["pen"]
            brush = curve.opts["brush"]
            logger.debug("Provided curve pen: {0}, brush: {1}".format(pen.color().name(), brush.color().name()))
        curve_new.setZValue(-100)
        vb.addItem(curve_new)

        self.curve_item_dict[name] = curve_new
        self.curve_vb_dict[name] = vb
        self.curve_ax_dict[name] = ax
        self.curve_group_list.append([name])
        self.highlighted_point_dict[name] = None
        self.setupData(name)

        curve_new.sigClicked.connect(self.setCurveFocus)
        # self.setCurveFocus(name)

        logger.debug("Calling updateViews")
        self.updateViews()
        return curve_new

    def removeCurve(self, name):
        logger.info("Removing curve {0}".format(name))
        # ind = self.curve_name_list.index(name)
        # vb = self.curve_vb_list.pop(ind)
        # ax = self.curve_ax_list.pop(ind)
        # curve = self.curve_list.pop(ind)
        vb = self.curve_vb_dict.pop(name)
        ax = self.curve_ax_dict.pop(name)
        curve = self.curve_item_dict.pop(name)
        # self.curve_name_list.pop(ind)
        pi = self.getPlotItem()
        pi.removeItem(vb)
        pi.removeItem(ax)
        pi.removeItem(curve)
        curve.setParent(None)
        ax.setParent(None)
        vb.setParent(None)
        curve.deleteLater()
        ax.deleteLater()
        vb.deleteLater()
        for gr_ind, group_name_list in enumerate(self.curve_group_list):
            if name in group_name_list:
                group_name_list.remove(name)
                if len(group_name_list) == 0:
                    self.curve_group_list.pop(gr_ind)

    def updateViews(self, updated_curve_name=None):
        t0 = time.time()
        pi = self.getPlotItem()
        # logger.info("Curve index {0} selected".format(self.curve_focus))
        if updated_curve_name is None:
            updated_curve_name = ""
        for group_name_list in self.curve_group_list:
            # logger.info("Curve group: {0}".format(group_name_list))
            for name in group_name_list:
                # Update x-axis of all curves
                vb = self.curve_vb_dict[name]
                vb.setGeometry(pi.vb.sceneBoundingRect())
                vb.linkedViewChanged(pi.vb, vb.XAxis)
                # Update y-axis of curve focus group and specified curve group
                if name == self.curve_focus:
                    for name2 in group_name_list:
                        vb2 = self.curve_vb_dict[name2]
                        vb2.linkedViewChanged(pi.vb, vb2.YAxis)
                        vr = vb2.viewRange()
                        self.update_curve_range_signal.emit(name2, vr[1][0], vr[1][1])
                elif name == updated_curve_name:
                    for name2 in group_name_list:
                        vb2 = self.curve_vb_dict[name2]
                        vb2.linkedViewChanged(vb, vb2.YAxis)
                        vr = vb2.viewRange()
                        self.update_curve_range_signal.emit(name2, vr[1][0], vr[1][1])
        # dt = time.time() - t0
        # logger.info("Updating view. {0:.1f} ms".format(dt * 1e3))

    def setCurveFocus(self, curve):
        if isinstance(curve, pg.GraphicsItem):
            curve = curve.name()
        try:
            curve_old = self.curve_item_dict[self.curve_focus]
            curve_old_pen: QtGui.QPen = curve_old.opts["pen"]
            curve_old_brush = curve_old.opts["symbolBrush"]
            logger.debug("Symbol brush: {0}".format(curve_old_brush))
            if curve_old_pen is not None:
                col = curve_old_pen.color()
                col.setAlphaF(self.unselected_pen_alpha)
                curve_old_pen.setColor(col)
                w = curve_old_pen.widthF()
                dw = self.selected_pen_factor
                curve_old_pen.setWidth(np.maximum(0, w - dw))
                curve_old.setPen(curve_old_pen)
            curve_old.setZValue(-100)

            if isinstance(curve_old_brush, list):
                for b in curve_old_brush:
                    col = b.color()
                    col.setAlphaF(self.unselected_pen_alpha)
                    b.setColor(col)
            else:
                try:
                    col = curve_old_brush.color()
                    col.setAlphaF(self.unselected_pen_alpha)
                    curve_old_brush.setColor(col)
                except AttributeError as e:
                    pass
            w = curve_old.opts["symbolSize"]
            dw = self.selected_pen_factor
            if isinstance(w, list):
                for s in w:
                    w -= dw
            else:
                w -= dw
            curve_old.opts["symbolBrush"] = curve_old_brush
            curve_old.opts["symbolSize"] = w
            curve_old.updateItems()

            vb_old = self.curve_vb_dict[self.curve_focus]
            vb_old.setZValue(-100)
        except KeyError:
            pass
        self.curve_focus = curve
        logger.debug("Curve {0} selected".format(self.curve_focus))
        pi = self.getPlotItem()
        axis_viewrange = self.curve_vb_dict[self.curve_focus].viewRange()
        logger.debug("Setting view range {0}".format(axis_viewrange[1]))
        pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
        pi_ax = pi.getAxis("right")

        curve_selected = self.curve_item_dict[self.curve_focus]
        curve_pen: QtGui.QPen = curve_selected.opts["pen"]
        curve_brush: QtGui.QBrush = curve_selected.opts["symbolBrush"]
        if curve_pen is not None:
            col = curve_pen.color()
            col.setAlphaF(self.selected_pen_alpha)
            curve_pen.setColor(col)
            w = curve_pen.widthF()
            if w > 0:
                dw = self.selected_pen_factor
            else:
                dw = 0
            curve_pen.setWidth(w + dw)
            curve_selected.setPen(curve_pen)

        if isinstance(curve_brush, list):
            for b in curve_brush:
                col = b.color()
                col.setAlphaF(self.unselected_pen_alpha)
                b.setColor(col)
        else:
            try:
                col = curve_brush.color()
                col.setAlphaF(self.unselected_pen_alpha)
                curve_brush.setColor(col)
            except AttributeError as e:
                pass
        w = curve_selected.opts["symbolSize"]
        dw = self.selected_pen_factor
        if isinstance(w, list):
            for s in w:
                s += dw
        else:
            w += dw
        curve_selected.opts["symbolSize"] = w
        curve_selected.opts["symbolBrush"] = curve_brush
        curve_selected.updateItems()

        if curve_pen is not None:
            pi_ax.setPen(curve_pen.color())
        else:
            pi_ax.setPen(curve_brush.color())
        pi.showGrid(True, True, 0.4)
        self.updateViews()

    def get_curve_range(self, curve_name):
        axis_viewrange = self.curve_vb_dict[curve_name].viewRange()
        return axis_viewrange

    def set_curve_color(self, curve_name, color):
        logger.debug("Set curve {0} color to {1}".format(curve_name, color))
        if curve_name == self.get_curve_focus_name():
            color.setAlphaF(self.selected_pen_alpha)
        else:
            color.setAlphaF(self.unselected_pen_alpha)
        ci: pg.PlotDataItem = self.curve_item_dict[curve_name]
        pen = ci.opts.get("pen")
        if pen is not None:
            pen.setColor(color)
            ci.setPen(pen)
        try:
            sym_brush = ci.opts["symbolBrush"]
            sym_brush.setColor(color)
            ci.opts["symbolBrush"] = sym_brush
            ci.opts["symbolPen"].setColor(color)
        except KeyError:
            logger.debug("No symbol brush")
        ci.updateItems()
        self.updateViews()

    def get_curve_color(self, curve_name) -> QtGui.QColor:
        logger.debug("Name: {0}".format(curve_name))
        logger.debug("Opts: {0}".format(self.curve_item_dict[curve_name].opts))
        pen = self.curve_item_dict[curve_name].opts["pen"]
        if pen is not None:
            curve_color = pen.color()
        else:
            curve_color = self.curve_item_dict[curve_name].opts["symbolBrush"].color()
        logger.debug("Curve color: {0}".format(curve_color))
        return curve_color

    def get_curve_focus_name(self):
        return self.curve_focus

    def get_curve(self, curve_name):
        curve = self.curve_item_dict[curve_name]
        return curve

    def showLegend(self, show_legend=True):
        if show_legend is True:
            if self.legend is None:
                self.legend = self.addLegend(offset=(5, 5))
                for it in self.curve_item_dict.values():
                    self.legend.addItem(it, it.opts.get('name', None))
        else:
            if self.legend is not None:
                self.legend.scene().removeItem(self.legend)
                self.legend = None

    def setCurveName(self, old_name, new_name):
        curve = self.curve_item_dict.pop(old_name)
        curve.opts["name"] = new_name
        self.curve_item_dict[new_name] = curve

    def set_curve_visible(self, name, visible):
        if visible:
            self.curve_item_dict[name].show()
        else:
            self.curve_item_dict[name].hide()
        self.updateViews()

    def addPoint(self, data, curve_name, auto_range=True):
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
        current_data_index = self.current_data_index_dict[curve_name]
        if current_data_index == 0:
            x_old = 0.0
        else:
            x_old = self.x_values_dict[curve_name][current_data_index]
        if (self.chronological is False) or (x_new > x_old):
            # Rescaling if the number of samples is too high
            if current_data_index + 1 >= self.values_size:
                current_data_index = int(self.values_size * 0.75)
                self.x_values_dict[curve_name][0:current_data_index] = self.x_values_dict[curve_name][self.values_size -
                                                                                                      current_data_index:
                                                                                                      self.values_size]
                self.y_values_dict[curve_name][0:current_data_index] = self.y_values_dict[curve_name][self.values_size -
                                                                                                      current_data_index:
                                                                                                      self.values_size]
            elif current_data_index == 0:
                self.x_values_dict[curve_name][0] = x_new
                self.y_values_dict[curve_name][0] = y_new
            current_data_index += 1
            self.x_values_dict[curve_name][current_data_index] = x_new
            start_index = np.argmax((self.x_values_dict[curve_name] - x_new) > -self.duration)
            self.y_values_dict[curve_name][self.current_data_index_dict[curve_name]] = y_new
            self.curve_item_dict[curve_name].setData(self.x_values_dict[curve_name][start_index:current_data_index] - x_new,
                                                     self.y_values_dict[curve_name][start_index:current_data_index],
                                                     antialias=False)
            if auto_range:
                vb = self.curve_vb_dict[curve_name]
                vb.enableAutoRange("y")
                vb.autoRange()
                if self.curve_focus == curve_name:
                    pi = self.getPlotItem()
                    axis_viewrange = self.curve_vb_dict[curve_name].viewRange()
                    # logger.debug("Setting view range {0}".format(axis_viewrange))
                    pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
                    pi.vb.setRange(xRange=axis_viewrange[0])
            self.current_data_index_dict[curve_name] = current_data_index
            t1 = time.time()
            # self.update()
            t2 = time.time()
            # logger.info("Add point timing: setup {0:.1f} ms, update {1:.1f} ms".format((t1-t0)*1e3, (t2-t1)*1e3))

    # def setData(self, x_data, y_data, curve_index=0, auto_range=True):
    def setData(self, x_data, y_data, curve_name, auto_range=True, **kargs):
        logger.debug("Setting data for curve {0}".format(curve_name))
        self.setupData(curve_name)
        n = x_data.shape[0]
        if n == 0:
            return
        self.x_values_dict[curve_name][-n:] = x_data
        self.y_values_dict[curve_name][-n:] = y_data
        vb = self.curve_vb_dict[curve_name]
        # vb.enableAutoRange("y")
        if auto_range:
            vb.enableAutoRange(pg.ViewBox.XYAxes, True)
        else:
            vb.enableAutoRange(pg.ViewBox.XYAxes, False)
        self.curve_item_dict[curve_name].setData(x_data, y_data, **kargs)
        if auto_range:
            vb.autoRange()
            for gr_list in self.curve_group_list:
                if curve_name in gr_list:
                    self.auto_range_group(gr_list)
        if self.curve_focus == curve_name:
            pi = self.getPlotItem()
            axis_viewrange = self.curve_vb_dict[curve_name].viewRange()
            logger.debug("Setting view range {0}".format(axis_viewrange))
            pi.vb.setRange(yRange=axis_viewrange[1], padding=0)
            pi.vb.setRange(xRange=axis_viewrange[0])
        self.updateViews(curve_name)

    def set_highlight(self, curve_name, point_index):

        pdi = self.curve_item_dict[curve_name]
        spi = pdi.scatter
        x = spi.data["size"]
        try:
            x[self.highlighted_point_dict[curve_name]] = self.symbol_size
            if point_index is not None:
                x[point_index] = self.highlight_symbol_size
        except IndexError as e:
            return
        pdi.opts['symbolSize'] = x
        pdi.updateItems()
        self.highlighted_point_dict[curve_name] = point_index

    def autoScale(self, curve_name=None):
        if curve_name is not None:
            for gr in self.curve_group_list:
                if curve_name in gr:
                    for cn in gr:
                        vb = self.curve_vb_dict[cn]
                        child_range = vb.childrenBounds(frac=[1.0, 1.0])
                        vb.setRange(yRange=child_range[1])
            self.updateViews()
        else:
            for vb in self.curve_vb_dict.values():
                child_range = vb.childrenBounds(frac=[1.0, 1.0])
                vb.setRange(yRange=child_range[1])

    def scaleAll(self, sc, center=None):
        for name, vb in self.curve_vb_dict.items():
            if center is None:
                vr = vb.targetRect()
                center = vr.center().y()
            top = center + sc * (vr.top() - center)
            bottom = center + sc * (vr.bottom() - center)
            logger.debug("Curve {0}: vr {1}, center {2}, top {3}, bottom {4}".format(name, vr, center, top, bottom))
            self.setYRange(top, bottom, padding=0)

    def auto_range_all(self):
        logger.debug("Auto ranging all curves.")
        pi = self.getPlotItem()
        x_min = np.inf
        x_max = -np.inf
        for name, vb in self.curve_vb_dict.items():
            vb.autoRange(padding=0.05)
            vr = vb.viewRange()
            x_min = np.minimum(x_min, vr[0][0])
            x_max = np.maximum(x_max, vr[0][1])
            self.update_curve_range_signal.emit(name, vr[1][0], vr[1][1])
            if name == self.curve_focus:
                pi.vb.setRange(yRange=vr[1], padding=0.1)
        pi.vb.setRange(xRange=[x_min, x_max], padding=0.05)

    def auto_range_group(self, curve_group):
        y_range = [np.inf, -np.inf]
        for cn in curve_group:
            vb = self.curve_vb_dict[cn]
            child_range = vb.childrenBounds(frac=[1.0, 1.0])
            try:
                y_range[0] = np.minimum(child_range[1][0], y_range[0])
                y_range[1] = np.maximum(child_range[1][1], y_range[1])
            except TypeError:
                pass

        if y_range[0] == np.inf:
            y_range[0] = 0
        if y_range[1] == -np.inf:
            y_range[1] = 1
        for cn in curve_group:
            logger.debug("Curve {0} setting range {1}".format(cn, y_range))
            vb = self.curve_vb_dict[cn]
            vb.setRange(yRange=y_range)
            self.update_curve_range_signal.emit(cn, y_range[0], y_range[1])

    def set_y_link(self, curve_1, curve_2):
        """
        Link y-axis of two curves.

        :param curve_1: name of first curve
        :param curve_2: name of second curve
        :param enable: set or clear link
        :return:
        """
        logger.debug("Curve group list: {0}".format(self.curve_group_list))
        old_group = list()
        for gr_ind, gr_list in enumerate(self.curve_group_list):
            if curve_1 in gr_list:
                gr_list.remove(curve_1)
                old_group = gr_list
                if len(gr_list) == 0:
                    self.curve_group_list.pop(gr_ind)
        found = False
        new_group = list()
        for gr_list in self.curve_group_list:
            if curve_2 in gr_list:
                gr_list.append(curve_1)
                found = True
                new_group = gr_list
                break
        if not found:
            self.curve_group_list.append([curve_1])
            new_group = [curve_1]
        self.auto_range_group(gr_list)
        return old_group, new_group

    def get_link_group(self, curve_name):
        """
        Return a list of curve names of the group that contains curve_name

        :param curve_name:
        :return:
        """
        for gr_list in self.curve_group_list:
            if curve_name in gr_list:
                return gr_list
        return None

    def stack_vertically(self):
        """
        Stack curves vertically to separate them
        :return:
        """
        dr = 1.0 / len(self.curve_group_list)
        r_gap = 0.05
        for ind, curve_group in enumerate(self.curve_group_list[::-1]):
            vr = [dr * ind + r_gap / 2, dr * (ind + 1) - r_gap / 2]
            y_range = [np.inf, -np.inf]
            for cn in curve_group:
                vb = self.curve_vb_dict[cn]
                child_range = vb.childrenBounds(frac=[1.0, 1.0])
                try:
                    y_range[0] = np.minimum(child_range[1][0], y_range[0])
                    y_range[1] = np.maximum(child_range[1][1], y_range[1])
                except TypeError:
                    pass
            if y_range[0] == np.inf:
                y_range[0] = 0
            if y_range[1] == -np.inf:
                y_range[1] = 1
            k = (y_range[1] - y_range[0]) / (vr[1] - vr[0])
            m = y_range[0] - k * vr[0]
            y_min = m
            y_max = k + m
            for cn in curve_group:
                logger.debug("Curve {0} setting range {1}-{2}".format(cn, y_min, y_max))
                vb = self.curve_vb_dict[cn]
                vb.setRange(yRange=[y_min, y_max])
                self.update_curve_range_signal.emit(cn, y_min, y_max)

    def points_clicked(self, curve, points):
        points = list(points)
        ind = [p.index() for p in points]
        pos = [curve.mapToScene(p.pos().toQPoint()) for p in points]
        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        logger.debug("Points on {0}: {1}, {2}, mouse {3}".format(curve.name(), ind, pos, mouse_pos))
        for p in points:
            pos = curve.mapToScene(p.pos().toQPoint())
            dist = (pos.x() - mouse_pos.x())**2 + (pos.y() - mouse_pos.y())**2
            p.dist = dist
        self.points_clicked_signal.emit(curve.name(), points)


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
            x_data = np.linspace(-600, 0, 50)
            y_data = np.sin(2*np.pi*x_data/240.0 * (c + 1)) + 10 * c
            name = "Curve {0}".format(c + 1)
            strip_tool.add_curve(name, symbol="t", width=None)
            strip_tool.set_data(x_data, y_data, name)
            # strip_tool.curve_vb_list[c].setRange(yRange=[c-1, c+1])
        # strip_tool.set_legend_position("bottom")
        strip_tool.set_y_link("Curve 1", "Curve 2")
        for c in range(5):
            x_data = np.linspace(-10, 10, 1000)
            y_data = x_data**(c % 3) + np.random.random(x_data.shape)
            name = "Curve {0}".format(c + 1)
            strip_tool2.add_curve(name, width=2)
            strip_tool2.set_data(x_data, y_data, name)
        strip_tool2.remove_curve("Curve 2")
        strip_tool2.add_curve("Curve 2")
        strip_tool2.set_y_link("Curve 3", "Curve 1")
        strip_tool2.remove_curve("Curve 4")
        strip_tool2.plot_widget.stack_vertically()
        strip_tool.plot_widget.set_highlight("Curve 1", 10)
        strip_tool.plot_widget.set_highlight("Curve 1", 15)
        # strip_tool.plot_widget.set_highlight("Curve 1", None)

    elif test == "trend":
        test_stream = TestStream()
        test_stream.show()

    sys.exit(app.exec_())
