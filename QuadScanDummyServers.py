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
from multiquad_lu import QuadSimulator

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
                                   memorized=True,
                                   hw_memorized=True,
                                   doc="Magnetic field", )

    position = attribute(label='position',
                         dtype=float,
                         access=pt.AttrWriteType.READ_WRITE,
                         unit="m",
                         format="%4.3f",
                         min_value=-100.0,
                         max_value=300.0,
                         fget="get_position",
                         fset="set_position",
                         memorized=True,
                         hw_memorized=True,
                         doc="Quad position in linac", )

    ql = attribute(label='length',
                   dtype=float,
                   access=pt.AttrWriteType.READ,
                   unit="m",
                   format="%4.3f",
                   min_value=-100.0,
                   max_value=300.0,
                   fget="get_ql",
                   memorized=False,
                   hw_memorized=False,
                   doc="Quad length", )

    energy = attribute(label='energy',
                   dtype=float,
                   access=pt.AttrWriteType.READ,
                   unit="MeV",
                   format="%4.3f",
                   min_value=0.0,
                   max_value=3000.0,
                   fget="get_energy",
                   memorized=False,
                   hw_memorized=False,
                   doc="Electron energy at quad", )

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
                           )

    qe = device_property(dtype=float,
                         doc="energy",
                         default_value=240)

    def __init__(self, klass, name):
        self.mainfieldcomponent_data = 0.0
        self.position_data = 0.0
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_mainfieldcomponent(self):
        return self.mainfieldcomponent_data + np.random.rand() * 0.001

    def set_mainfieldcomponent(self, k):
        self.info_stream("In set_mainfieldcomponent: New k={0:.3f}".format(k))
        self.mainfieldcomponent_data = k
        return True

    def get_position(self):
        return self.position_data

    def set_position(self, new_pos):
        self.info_stream("In set_position: New position {0} m".format(new_pos))
        self.position_data = new_pos

    def get_ql(self):
        return self.length

    def get_energy(self):
        return self.qe


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

    position = attribute(label='position',
                         dtype=float,
                         access=pt.AttrWriteType.READ_WRITE,
                         unit="m",
                         format="%4.3f",
                         min_value=-100.0,
                         max_value=300.0,
                         fget="get_position",
                         fset="set_position",
                         memorized=True,
                         hw_memorized=True,
                         doc="Position in linac", )

    def __init__(self, klass, name):
        self.scr_in_state = False
        self.position_data = 0.0
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

    def get_position(self):
        return self.position_data

    def set_position(self, new_pos):
        self.info_stream("In set_position: New position {0} m".format(new_pos))
        self.position_data = new_pos


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
                          access=pt.AttrWriteType.READ_WRITE,
                          display_level=pt.DispLevel.OPERATOR,
                          unit="Hz",
                          format="%5.2f",
                          fget="get_framerate",
                          fset="set_framerate",
                          memorized=True,
                          hw_memorized=True,
                          doc="Camera framerate", )

    pixel_cal = attribute(label='Pixel calibration',
                          dtype=float,
                          access=pt.AttrWriteType.READ,
                          display_level=pt.DispLevel.OPERATOR,
                          unit="m",
                          format="%5.2f",
                          fget="get_pixel_cal",
                          doc="Resolution of camera pixel", )

    roi = attribute(label='Region of interest',
                    dtype=[int],
                    access=pt.AttrWriteType.READ_WRITE,
                    display_level=pt.DispLevel.OPERATOR,
                    max_dim_x=4,
                    unit="px",
                    format="%d",
                    fget="get_roi",
                    fset="set_roi",
                    doc="Region of interest", )

    position = attribute(label='position',
                         dtype=float,
                         access=pt.AttrWriteType.READ_WRITE,
                         unit="m",
                         format="%4.3f",
                         min_value=-100.0,
                         max_value=300.0,
                         fget="get_position",
                         fset="set_position",
                         memorized=True,
                         hw_memorized=True,
                         doc="Position in linac", )

    noiselevel = attribute(label='noiselevel',
                           dtype=float,
                           access=pt.AttrWriteType.READ_WRITE,
                           unit="counts",
                           format="%4.3f",
                           min_value=0.0,
                           max_value=3000.0,
                           fget="get_noiselevel",
                           fset="set_noiselevel",
                           memorized=True,
                           hw_memorized=True,
                           doc="Image background noise level", )

    charge = attribute(label='charge',
                       dtype=float,
                       access=pt.AttrWriteType.READ_WRITE,
                       unit="pC",
                       format="%4.3f",
                       min_value=-100.0,
                       max_value=1000.0,
                       fget="get_charge",
                       fset="set_charge",
                       memorized=True,
                       hw_memorized=True,
                       doc="Beam total charge", )

    alpha = attribute(label='alpha',
                      dtype=float,
                      access=pt.AttrWriteType.READ_WRITE,
                      unit="",
                      format="%4.3f",
                      min_value=-1000.0,
                      max_value=1000.0,
                      fget="get_alpha",
                      fset="set_alpha",
                      memorized=True,
                      hw_memorized=True,
                      doc="Electron beam alpha", )

    beta = attribute(label='beta',
                     dtype=float,
                     access=pt.AttrWriteType.READ_WRITE,
                     unit="m",
                     format="%4.3f",
                     min_value=-100000.0,
                     max_value=3000000.0,
                     fget="get_beta",
                     fset="set_beta",
                     memorized=True,
                     hw_memorized=True,
                     doc="Electron beam beta", )

    eps_n = attribute(label='eps_n',
                      dtype=float,
                      access=pt.AttrWriteType.READ_WRITE,
                      unit="um",
                      format="%4.3f",
                      min_value=0.0,
                      max_value=1000.0,
                      fget="get_eps",
                      fset="set_eps",
                      memorized=True,
                      hw_memorized=True,
                      doc="Electron beam emittance", )

    alpha_y = attribute(label='alpha y',
                        dtype=float,
                        access=pt.AttrWriteType.READ_WRITE,
                        unit="",
                        format="%4.3f",
                        min_value=-1000.0,
                        max_value=1000.0,
                        fget="get_alpha_y",
                        fset="set_alpha_y",
                        memorized=True,
                        hw_memorized=True,
                        doc="Electron beam alpha vertical", )

    beta_y = attribute(label='beta y',
                       dtype=float,
                       access=pt.AttrWriteType.READ_WRITE,
                       unit="m",
                       format="%4.3f",
                       min_value=-100000.0,
                       max_value=3000000.0,
                       fget="get_beta_y",
                       fset="set_beta_y",
                       memorized=True,
                       hw_memorized=True,
                       doc="Electron beam beta vertical", )

    eps_n_y = attribute(label='eps_n y',
                        dtype=float,
                        access=pt.AttrWriteType.READ_WRITE,
                        unit="um",
                        format="%4.3f",
                        min_value=0.0,
                        max_value=1000.0,
                        fget="get_eps_y",
                        fset="set_eps_y",
                        memorized=True,
                        hw_memorized=True,
                        doc="Electron beam emittance vertical", )

    beamenergy = attribute(label='beam energy',
                           dtype=float,
                           access=pt.AttrWriteType.READ_WRITE,
                           unit="MeV",
                           format="%4.3f",
                           min_value=0.0,
                           max_value=6000.0,
                           fget="get_beamenergy",
                           fset="set_beamenergy",
                           memorized=True,
                           hw_memorized=True,
                           doc="Electron beam energy", )


    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    section = device_property(dtype=str,
                              doc="Section",
                              default_value="MS1")

    magnet_names = device_property(dtype=(str, ),
                                   doc="Magnet devices to simulate for this section",
                                   default_value=("i-ms1/mag/qb-01", "i-ms1/mag/qb-02", "i-ms1/mag/qb-03",
                                                  "i-ms1/mag/qb-04"))
    screen_name = device_property(dtype=str,
                                  doc="Screen device to simulate for this section",
                                  default_value="i-ms1/dia/scrn-01")

    def __init__(self, klass, name):
        self.image_data = np.zeros((1280, 1024))
        self.width = 1280
        self.height = 1024
        self.px = 15e-6
        self.roi_data = np.array([0, 0, 1280, 1024])
        self.charge_data = 100.0
        self.position_data = 0.0
        self.framerate_data = 2.0
        # self.magnet_names = ["i-ms1/mag/qb-01", "i-ms1/mag/qb-02", "i-ms1/mag/qb-03", "i-ms1/mag/qb-04"]
        # self.screen_name = "i-ms1/dia/scrn-01"
        self.gamma_energy = 233e6 / 0.511e6
        self.alpha_data = 10.0
        self.beta_data = 27.0
        self.eps_n_data = 2
        self.alpha_y_data = -5.0
        self.beta_y_data = 20.0
        self.eps_n_y_data = 1.5
        self.noiselevel_data = 20.0
        self.sim = QuadSimulator(self.alpha_data, self.beta_data, self.eps_n_data * 1e-6 / self.gamma_energy,
                                 self.alpha_y_data, self.beta_y_data, self.eps_n_y_data * 1e-6/ self.gamma_energy,
                                 add_noise=True)  # MS-1 QB-01, SCRN-01
        self.magnet_devices = list()
        self.screen_device = None
        self.running = False
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        for mag in self.magnet_names:
            dev = pt.DeviceProxy("127.0.0.1:10000/{0}#dbase=no".format(mag))
            self.magnet_devices.append(dev)
            pos = dev.position
            self.sim.add_quad(SectionQuad(mag, pos, 0.2, "MAG-01", "CRQ-01", False))
            self.debug_stream("Connected to device {0}".format(mag))
        self.screen_device = pt.DeviceProxy("127.0.0.1:10001/{0}#dbase=no".format(self.screen_name))
        self.debug_stream("Connected to device {0}".format(self.screen_name))
        self.info_stream("Attributes: {0}".format(self.screen_device.get_attribute_list()))
        pos = self.screen_device.position
        self.sim.set_screen_position(pos)

        self.info_stream("Section {0}".format(self.section))
        self.info_stream("Magnets: {0}".format(self.magnet_names))

        self.debug_stream("init_device finished")

    def update_sim(self):
        self.sim.alpha = self.alpha_data
        self.sim.beta = self.beta_data
        self.sim.eps = self.eps_n_data * 1e-6 / self.gamma_energy
        self.sim.alpha_y = self.alpha_y_data
        self.sim.beta_y = self.beta_y_data
        self.sim.eps_y = self.eps_n_y_data * 1e-6 / self.gamma_energy

    def get_image(self):
        if self.running:
            k_list = list()
            for dev in self.magnet_devices:
                k_list.append(dev.mainfieldcomponent)
            sigma_x = self.sim.get_screen_beamsize(k_list, "x")
            sigma_y = self.sim.get_screen_beamsize(k_list, "y")
            x = self.px * (np.arange(self.width) - self.width / 2)
            y = self.px * (np.arange(self.height) - self.height / 2)
            X, Y = np.meshgrid(x, y)
            self.debug_stream("Beamsize: {0:.3f} x {1:.3f} mm".format(sigma_x * 1e3, sigma_y * 1e3))
            beam_image = 2e-6 * self.charge_data / sigma_x / sigma_y * np.exp(-X**2/(2*sigma_x**2)) * np.exp(-Y**2/(2*sigma_y**2))
            self.image_data = np.minimum((beam_image + self.noiselevel_data * np.random.random((self.height, self.width))).astype(np.uint16), 4096)
            n_noise = np.maximum(0, 500 + self.noiselevel_data * np.random.normal(50, 25, 1)).astype(int)
            xr = (self.width * np.random.random(n_noise)).astype(int)
            yr = (self.height * np.random.random(n_noise)).astype(int)
            ir = np.maximum(0, np.minimum(4096, np.random.normal(2560, 512, n_noise))).astype(np.uint16)
            self.image_data[yr, xr] = ir
        return self.image_data

    def get_framerate(self):
        return self.framerate_data

    def set_framerate(self, data):
        self.info_stream("In set_framerate: New framerate {0} Hz".format(data))
        self.framerate_data = data

    def get_pixel_cal(self):
        return self.px

    def get_roi(self):
        return self.roi_data

    def set_roi(self, value):
        self.roi_data = value

    def get_position(self):
        return self.position_data

    def set_position(self, new_pos):
        self.info_stream("In set_position: New position {0} m".format(new_pos))
        self.position_data = new_pos

    def get_noiselevel(self):
        return self.noiselevel_data

    def set_noiselevel(self, value):
        self.noiselevel_data = value

    def get_charge(self):
        return self.charge_data

    def set_charge(self, value):
        self.charge_data = value

    def get_alpha(self):
        return self.alpha_data

    def set_alpha(self, value):
        self.alpha_data = value
        self.update_sim()

    def get_beta(self):
        return self.beta_data

    def set_beta(self, value):
        self.beta_data = value
        self.update_sim()

    def get_eps(self):
        return self.eps_n_data

    def set_eps(self, value):
        self.eps_n_data = value
        self.update_sim()

    def get_alpha_y(self):
        return self.alpha_y_data

    def set_alpha_y(self, value):
        self.alpha_y_data = value
        self.update_sim()

    def get_beta_y(self):
        return self.beta_y_data

    def set_beta_y(self, value):
        self.beta_y_data = value
        self.update_sim()

    def get_eps_y(self):
        return self.eps_n_y_data

    def set_eps_y(self, value):
        self.eps_n_y_data = value
        self.update_sim()

    def get_beamenergy(self):
        return self.gamma_energy * 0.511

    def set_beamenergy(self, value):
        self.gamma_energy = value / 0.511
        self.update_sim()

    @command
    def start(self):
        self.info_stream("Starting camera {0}".format(self.screen_name))
        self.running = True
        self.set_state(pt.DevState.RUNNING)

    @command
    def stop(self):
        self.info_stream("Stopping camera {0}".format(self.screen_name))
        self.running = False
        self.set_state(pt.DevState.ON)


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

    position = attribute(label='position',
                         dtype=float,
                         access=pt.AttrWriteType.READ_WRITE,
                         unit="m",
                         format="%4.3f",
                         min_value=-100.0,
                         max_value=300.0,
                         fget="get_position",
                         fset="set_position",
                         memorized=True,
                         hw_memorized=True,
                         doc="Position in linac", )

    roi = attribute(label='roi',
                    dtype=[float],
                    access=pt.AttrWriteType.READ,
                    max_dim_x=4,
                    unit="m",
                    format="%4.3f",
                    min_value=-2048.0,
                    max_value=2048.0,
                    fget="get_roi",
                    doc="ROI for calibration", )

    # --- Device properties
    #
    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=10.0)

    def __init__(self, klass, name):
        self.cal = "{\"angle\": 0.0, \"pos\": [258.5, 48.57], \"size\": [357.0, 357.0]}"
        self.position_data = 0.0
        self.width = 5.0
        self.roi_data = [0, 200, 0, 200]
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

    def get_position(self):
        return self.position_data

    def set_position(self, new_pos):
        self.info_stream("In set_position: New position {0} m".format(new_pos))
        self.position_data = new_pos

    def get_roi(self):
        return self.roi_data


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

    import copy
    args_0 = copy.copy(args)
    args_0.append("test")
    args_0.append("-file=./DummyServerConfig/dummymagnet.cfg")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10000")
    args_0.append("-v4")
    p = multiprocessing.Process(target=run_class, args=[DummyMagnet, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = copy.copy(args)
    args_0.append("test")
    args_0.append("-file=./DummyServerConfig/dummyscreen.cfg")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10001")
    args_0.append("-v4")
    p = multiprocessing.Process(target=run_class, args=[DummyScreen, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = copy.copy(args)
    args_0.append("test")
    args_0.append("-file=./DummyServerConfig/dummybeamviewer.cfg")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10003")
    args_0.append("-v4")
    p = multiprocessing.Process(target=run_class, args=[DummyBeamviewer, args_0])
    p.start()
    p_list.append(p)

    port0 += 1
    args_0 = copy.copy(args)
    args_0.append("test")
    args_0.append("-file=./DummyServerConfig/dummylimaccd.cfg")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10004")
    args_0.append("-v4")
    p = multiprocessing.Process(target=run_class, args=[DummyLimaccd, args_0])
    p.start()
    p_list.append(p)

    time.sleep(0.5)
    port0 += 1
    args_0 = copy.copy(args)
    args_0.append("test")
    args_0.append("-file=./DummyServerConfig/dummyliveviewer.cfg")
    args_0.append("-ORBendPoint")
    args_0.append("giop:tcp::10002")
    args_0.append("-v4")
    p = multiprocessing.Process(target=run_class, args=[DummyLiveviewer, args_0])
    p.start()
    p_list.append(p)

    # port0 += 1
    # args_0 = copy.copy(args)
    # args_0.append("-dlist")
    # args_0.append("i-ms1/dia/scrn-01")
    # args_0.append("-ORBendPoint")
    # args_0.append("giop:tcp::10001")
    # p = multiprocessing.Process(target=run_class, args=[DummyScreen, args_0])
    # p.start()
    # p_list.append(p)
    #
    # port0 += 1
    # args_0 = copy.copy(args)
    # args_0.append("-dlist")
    # args_0.append("lima/beamviewer/i-ms1-dia-scrn-01")
    # args_0.append("-ORBendPoint")
    # args_0.append("giop:tcp::10002")
    # p = multiprocessing.Process(target=run_class, args=[DummyBeamviewer, args_0])
    # p.start()
    # p_list.append(p)
    #
    # port0 += 1
    # args_0 = copy.copy(args)
    # args_0.append("-dlist")
    # args_0.append("lima/liveviewer/i-ms1-dia-scrn-01")
    # args_0.append("-ORBendPoint")
    # args_0.append("giop:tcp::10003")
    # p = multiprocessing.Process(target=run_class, args=[DummyLiveviewer, args_0])
    # p.start()
    # p_list.append(p)
    #
    # port0 += 1
    # args_0 = copy.copy(args)
    # args_0.append("-dlist")
    # args_0.append("lima/limaccd/i-ms1-dia-scrn-01")
    # args_0.append("-ORBendPoint")
    # args_0.append("giop:tcp::10004")
    # p = multiprocessing.Process(target=run_class, args=[DummyLimaccd, args_0])
    # p.start()
    # p_list.append(p)

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

