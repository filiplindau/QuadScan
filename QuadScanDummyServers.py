"""
Created 2019-03-21

Dummy device servers for testing scans without actual devices.

@author: Filip Lindau
"""

import threading
import multiprocessing
import uuid
import logging
import time
import inspect
import PIL
import numpy as np
from numpngw import write_png
import os
from collections import namedtuple
import pprint
import traceback
from scipy.signal import medfilt2d

from tasks.GenericTasks import *
from QuadScanDataStructs import *


try:
    import PyTango as pt
    from PyTango.server import Device, DeviceMeta
    from PyTango.server import attribute, command
    from PyTango.server import device_property
except ImportError:
    try:
        import tango as pt
        from tango.server import Device, DeviceMeta
        from tango.server import attribute, command
        from tango.server import device_property
    except ModuleNotFoundError:
        pass

logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class DummyMagnet(Device):
    __metaclass__ = DeviceMeta

    # --- Operator attributes
    #
    mainfieldcomponent_data = attribute(label='mainfieldcomponent',
                                        dtype=float,
                                        access=pt.AttrWriteType.READ_WRITE,
                                        unit="k",
                                        format="%4.3f",
                                        min_value=-100.0,
                                        max_value=100.0,
                                        fget="get_mainfieldcomponent",
                                        fset="set_mainfieldcomponent",
                                        memorized=True,
                                        hw_memorized=False,
                                        doc="Magnetic field", )

    # --- Device properties
    #
    length = device_property(dtype=float,
                             doc="Quad length",
                             default_value=0.2)

    polarity = device_property(dtype=int,
                               doc="Polarity",
                               default_value=1)

    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=5.0)

    def __init__(self, klass, name):
        self.mainfieldcomponent_data = 0.0
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_mailfieldcomponent(self):
        return self.mainfieldcomponent_data

    def set_mainfieldcomponent(self, k):
        self.mainfieldcomponent_data = k
        return True


class DummyScreen(Device):
    __metaclass__ = DeviceMeta

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    # --- Operator attributes
    #
    statusin = attribute(label='statusin',
                         dtype=bool,
                         access=pt.AttrWriteType.READ,
                         unit="",
                         fget="get_statusin",
                         memorized=False,
                         hw_memorized=False,
                         doc="Screen in status", )

    statusout = attribute(label='statusout',
                         dtype=bool,
                         access=pt.AttrWriteType.READ,
                         unit="",
                         fget="get_statusout",
                         memorized=False,
                         hw_memorized=False,
                         doc="Screen out status", )

    def __init__(self, klass, name):
        self.scr_in_state = False
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.CLOSE)

        self.debug_stream("init_device finished")

    @command
    def movein(self):
        self.info_stream("Closing screen")
        self.scr_in_state = True
        self.set_state(pt.DevState.EXTRACT)

    @command
    def moveout(self):
        self.info_stream("Open screen")
        self.scr_in_state = False
        self.set_state(pt.DevState.OPEN)

    def get_statusin(self):
        return self.scr_in_state

    def get_statusout(self):
        return not self.scr_in_state


class DummyLiveviewer(Device):
    __metaclass__ = DeviceMeta

    # --- Operator attributes
    #
    image_data = attribute(label='image',
                           dtype=[[np.double]],
                           access=pt.AttrWriteType.READ,
                           max_dim_x=2048,
                           max_dim_y=2048,
                           display_level=pt.DispLevel.OPERATOR,
                           unit="a.u.",
                           format="%5.2f",
                           fget="get_image",
                           doc="Camera image", )

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        self.image_data = np.zeros((1280, 1024))
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_image(self):
        self.image_data = np.random.random_integers(0, 2 ** 16, (1280, 1024)).astype(np.uint16)
        return self.image_data


class DummyBeamviewer(Device):
    __metaclass__ = DeviceMeta

    # --- Operator attributes
    #
    measurementruler = attribute(label='measurementruler',
                                 dtype=str,
                                 access=pt.AttrWriteType.READ,
                                 unit="",
                                 format="%s",
                                 fget="get_measurementruler",
                                 memorized=True,
                                 hw_memorized=False,
                                 doc="Measurement ruler", )

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        self.cal = "{\"angle\": 0.0, \"pos\": [258.5, 48.57], \"size\": [688.13, 702.02]}"
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_measurementruler(self):
        return self.cal
