"""
Created 2018-10-24

Handling of tango device attribute access.

@author: Filip Lindau
"""

import logging
import threading
from twisted_cut.defer import Deferred
from TangoTwisted import deferred_from_future
from TangoTwisted import defer_to_thread
try:
    import PyTango.futures as tangof
except ImportError:
    import tango.futures as tangof


class TangoDeviceEngine(object):
    """
    Class that handles all access to tango devices and attributes.

    Cointains a dict with opened devices. Manages read and writes
    of attributed to these devices. Optionally issues looping calls
    to read attribute. When requesting a new attribute the attribute
    config is pulled initially.

    If an attribute is requested from a device that is not loaded
    that device is first loaded and then the attribute read.
    """
    def __init__(self):
        self.logger = logging.getLogger("TangoDeviceEngine")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("TangoDeviceEngine.__init__")

        self.device_dict = dict()
        self.waiting_attrs = dict()
        self.attr_configs = dict()

        self.lock = threading.Lock()

    def read_attribute(self, attr_name, device_name, loop_period=None, cached=False):
        if device_name not in self.device_dict:
            d = self.add_device(device_name)
        else:
            d = Deferred()
            d.callback(self.device_dict[device_name])
        d.addCallback(defer_to_thread())

    def write_attribute(self, attr_name, device_name):
        pass

    def check_attribute(self, attr_name, device_name, target_value, period=0.3, timeout=1.0,
                        tolerance=None, write=True):
        pass

    def get_attribute_config(self, attr_name, device_name):
        pass

    def send_command(self, cmd_name, device_name, cmd_data=None):
        pass

    def add_device(self, device_name):
        if device_name in self.device_dict:
            self.logger.info("Device {0} already in dict".format(device_name))
            dev = self.device_dict[device_name]
        else:
            self.logger.info("Adding device {0}".format(device_name))
            d = deferred_from_future(tangof.DeviceProxy(device_name))
            d.addCallbacks(self._add_device_cb, self._add_device_eb)
        return dev

    def remove_device(self, device_name):
        pass

    def add_attribute(self, attr_name, device_name):
        pass

    def _attr_error_cb(self, err):
        self.logger.error("Error from attribute: {0}".format(err))
        return err

    def _attr_read_cb(self, result):
        self.logger.debug("Read attribute returned {0}".format(result))

    def _cancel_device(self):
        pass

    def _add_device_cb(self, result):
        self.logger.info("Device proxy received. Adding {0} to dict.".format(result.name()))
        with self.lock:
            self.device_dict[result.name()] = result
        return True

    def _add_device_eb(self, err):
        self.logger.error("Could not add device to dict: {0}".format(err))
        return err

    def _get_attr_config(self, attr_name, device_name):
        if device_name not in self.device_dict:
            self.add_device(device_name)
        dev = self.device_dict[device_name]
        full_name = "{0}/{1}".format(device_name, attr_name)
        attr_config = dev.get_attribute_config(attr_name)
        with self.lock:
            self.attr_configs[full_name] = attr_config
