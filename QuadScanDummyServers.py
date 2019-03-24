"""
Created 2019-03-21

Dummy device servers for testing scans without actual devices.

@author: Filip Lindau
"""

import threading
import multiprocessing
import logging
import time
import sys

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

logger = logging.getLogger("DummyServers")
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
    mainfieldcomponent = attribute(label='mainfieldcomponent',
                                   dtype=float,
                                   access=pt.AttrWriteType.READ_WRITE,
                                   unit="k",
                                   format="%4.3f",
                                   min_value=-100.0,
                                   max_value=100.0,
                                   fget="get_mainfieldcomponent",
                                   fset="set_mainfieldcomponent",
                                   memorized=False,
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
        logger.info("In DummeMagnet: {0} {1}".format(klass, name))
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_mainfieldcomponent(self):
        return self.mainfieldcomponent_data + np.random.rand() * 0.01

    def set_mainfieldcomponent(self, k):
        self.mainfieldcomponent_data = k
        return True

    # def device_name_factory(self, list):
    #     self.info_stream("Adding server MS1/MAG/QF01")
    #     list.append("MS1/MAG/QF01")
    #
    #     return list


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
    image = attribute(label='image',
                           dtype=[[np.double]],
                           access=pt.AttrWriteType.READ,
                           max_dim_x=2048,
                           max_dim_y=2048,
                           display_level=pt.DispLevel.OPERATOR,
                           unit="a.u.",
                           format="%5.2f",
                           fget="get_image",
                           doc="Camera image", )

    framerate = attribute(label='framerate',
                           dtype=float,
                           access=pt.AttrWriteType.READ,
                           display_level=pt.DispLevel.OPERATOR,
                           unit="Hz",
                           format="%5.2f",
                           fget="get_framerate",
                           doc="Camera framerate", )

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        self.image_data = np.zeros((1280, 1024))
        self.framerate_data = 2.0
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_image(self):
        self.image_data = np.random.random_integers(0, 2 ** 16, (800, 600)).astype(np.uint16)
        return self.image_data

    def get_framerate(self):
        return self.framerate_data


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
                                 memorized=False,
                                 hw_memorized=False,
                                 doc="Measurement ruler", )

    measurementrulerwidth = attribute(label='measurementrulerwidth',
                                 dtype=float,
                                 access=pt.AttrWriteType.READ,
                                 unit="",
                                 format="%s",
                                 fget="get_measurementrulerwidth",
                                 memorized=False,
                                 hw_memorized=False,
                                 doc="Measurement ruler", )

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        self.cal = "{\"angle\": 0.0, \"pos\": [258.5, 48.57], \"size\": [688.13, 702.02]}"
        self.width = 20.0
        logger.info("In DummyBeamViewer: {0} {1}".format(klass, name))
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_measurementruler(self):
        return self.cal

    def get_measurementrulerwidth(self):
        return self.width


class DummyLimaccd(Device):
    __metaclass__ = DeviceMeta

    # --- Operator attributes
    #

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        logger.info("In DummyLimaccd: {0} {1}".format(klass, name))
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")


def run_class(cl_inst, args):
    pt.server.server_run((cl_inst,), args=args)


if __name__ == "__main__":

    # Start with: python .\QuadDummyServers.py test -nodb -v4

    args = sys.argv
    logger.info("Args: {0}".format(args))
    port0 = 10000

    p_list = list()

    args_0 = args.copy()
    args_0.append("-dlist")
    args_0.append("ms1/mag/qb-01")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10000")
    p = multiprocessing.Process(target=run_class, args=[DummyMagnet, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = args.copy()
    args_0.append("-dlist")
    args_0.append("i-ms1/dia/scrn-01")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10001")
    p = multiprocessing.Process(target=run_class, args=[DummyScreen, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = args.copy()
    args_0.append("-dlist")
    args_0.append("lima/beamviewer/i-ms1-dia-scrn-01")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10002")
    p = multiprocessing.Process(target=run_class, args=[DummyBeamviewer, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = args.copy()
    args_0.append("-dlist")
    args_0.append("lima/liveviewer/i-ms1-dia-scrn-01")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10003")
    p = multiprocessing.Process(target=run_class, args=[DummyLiveviewer, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = args.copy()
    args_0.append("-dlist")
    args_0.append("lima/limaccd/i-ms1-dia-scrn-01")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10004")
    p = multiprocessing.Process(target=run_class, args=[DummyLiveviewer, args_0])
    p.start()
    p_list.append(p)

    for p in p_list:
        p.join()
    # try:
    #     ind = args.index("-ds_class")
    #     ds_class = args[ind+1]
    #     args.pop(ind)
    #     args.pop(ind)
    # except (ValueError, IndexError):
    #     ds_class = None
    # if ds_class is not None:
    #     if ds_class == "DummyBeamviewer":
    #         pt.server.server_run((DummyBeamviewer,), args=args)
    #     elif ds_class == "DummyMagnet":
    #         pt.server.server_run((DummyMagnet,), args=args)
    #     elif ds_class == "DummyScreen":
    #         pt.server.server_run((DummyScreen,), args=args)
    #     elif ds_class == "DummyLiveviewer":
    #         pt.server.server_run((DummyLiveviewer,), args=args)
    # else:
    #     pt.server.server_run((DummyMagnet,))
    # pt.server.server_run((DummyBeamviewer, DummyLiveviewer, DummyMagnet, DummyScreen))

