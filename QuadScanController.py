# -*- coding:utf-8 -*-
"""
Created on May 18, 2018

@author: Filip Lindau
"""

import threading
import time
import logging
from twisted_cut import defer
from twisted_cut.failure import Failure
import TangoTwisted
from TangoTwisted import TangoAttributeFactory, defer_later
from collections import OrderedDict
try:
    import tango
except ImportError:
    import PyTango as tango
import QuadScanState as qs
import numpy as np
from scipy.signal import medfilt2d
from PyQt4 import QtCore
from multiprocessing import Pool

# logger = logging.getLogger("QuadScanController")
# while len(logger.handlers):
#     logger.removeHandler(logger.handlers[0])
#
# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
# f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
# fh = logging.StreamHandler()
# fh.setFormatter(f)
# logger.addHandler(fh)
# logger.setLevel(logging.DEBUG)


class QuadScanController(QtCore.QObject):
    progress_signal = QtCore.Signal(float)          # Signal for updating progress bar. Argument is completion 0-1
    processing_done_signal = QtCore.Signal()        # signal when all processing is done
    image_done_signal = QtCore.Signal(int, int)     # Emit when a single image is done processing. Image number as arg
    fit_done_signal = QtCore.Signal()
    state_change_signal = QtCore.Signal(str, str)   # State, Status strings
    attribute_ready_signal = QtCore.Signal(object)  # Returns with a pytango.Attribute
    load_parameters_signal = QtCore.Signal()        # Scan parameters loaded
    camera_roi_read_done_signal = QtCore.Signal()   # ROI read from camera
    quad_selected_signal = QtCore.Signal()          # New quad selected. Time to update electron energy, position, value

    def __init__(self, quad_name=None, screen_name=None, start=False):
        """
        Controller for running a quad scan. Communicates with a quad magnet and a camera.


        :param quad_name: Tango device name of the quad used for scanning
        :param screen_name: Tango device name of the camera used for capturing images of the screen
        :param start: If True, the tango device objects are created.
        """
        QtCore.QObject.__init__(self)
        self.device_names = dict()                  # A dict containing the names of connected devices
        if quad_name is not None:
            self.device_names["quad"] = quad_name
        if screen_name is not None:
            self.device_names["screen"] = screen_name

        self.device_factory_dict = dict()

        self.setup_attr_params = dict()
        # self.setup_attr_params["speed"] = ("motor", "speed", 50.0)

        self.looping_calls = list()

        # Dict for storing result from attribute reads
        self.attr_result = dict()
        self.attr_result["image"] = None
        self.attr_result["magnet"] = None

        # Dict for configuring attributes to read when idle
        self.idle_params = dict()
        self.idle_params["reprate"] = 2.0
        self.idle_params["camera_attr"] = "image"
        self.idle_params["paused"] = False

        # Dict for configuring scan
        # Also stores the available quads and screens in different sections
        self.scan_params = dict()
        self.scan_params["k_min"] = 8.6
        self.scan_params["k_max"] = 8.75
        self.scan_params["num_k_values"] = 1
        self.scan_params["num_shots"] = 1
        self.scan_params["quad_name"] = None
        self.scan_params["quad_device_names"] = dict()
        self.scan_params["quad_length"] = 1.0
        self.scan_params["quad_pos"] = 1.0
        self.scan_params["quad_screen_dist"] = 1.0
        self.scan_params["screen_name"] = None
        self.scan_params["screen_device_names"] = dict()
        self.scan_params["screen_pos"] = 1.0
        self.scan_params["section_name"] = "ms1"
        self.scan_params["pixel_size"] = [1.0, 1.0]
        self.scan_params["bpp"] = 16
        self.scan_params["electron_energy"] = 1.0
        self.scan_params["roi_center"] = [1.0, 1.0]
        self.scan_params["roi_dim"] = [1.0, 1.0]
        self.scan_params["sections"] = ["ms1", "ms2", "ms3", "sp02"]
        self.scan_params["section_quads"] = dict()      # Element is a list of quads, each quad a dict with keys
                                                        # name, position, length, crq, polarity
        self.scan_params["section_screens"] = dict()    # Element is a list of screens, each screen a dict with keys
                                                        # name, position
        # self.scan_params["dev_name"] = "motor"

        # Dict for storing result of a scan
        # When loading from disk the data is stored here also
        self.scan_result = dict()
        self.scan_result["k_data"] = None
        self.scan_result["raw_data"] = None
        self.scan_result["proc_data"] = None
        self.scan_result["scan_data"] = None
        self.scan_result["line_data_x"] = None
        self.scan_result["line_data_y"] = None
        self.scan_result["x_cent"] = None
        self.scan_result["y_cent"] = None
        self.scan_result["sigma_x"] = None
        self.scan_result["sigma_y"] = None
        self.scan_result["enabled_data"] = None
        self.scan_result["charge_data"] = None
        self.scan_result["fit_poly"] = None
        self.scan_result["start_time"] = None
        self.scan_raw_data = None
        self.scan_proc_data = None
        self.scan_roi_data = None

        # Dict for storing analysis results of emittance calculations
        self.analysis_result = dict()
        self.analysis_result["eps"] = None
        self.analysis_result["beta"] = None
        self.analysis_result["gamma"] = None
        self.analysis_result["alpha"] = None
        self.analysis_result["fit_data"] = None

        # Dict for configuring analysis parameters
        self.analysis_params = dict()
        self.analysis_params["fit_algo"] = "full_matrix"
        self.analysis_params["roi"] = "full"
        self.analysis_params["threshold"] = 0.001
        self.analysis_params["median_kernel"] = 3
        self.analysis_params["background_subtract"] = True
        self.analysis_params["quad_length"] = 1.0
        self.analysis_params["quad_screen_dist"] = 1.0
        self.analysis_params["electron_energy"] = 1.0

        self.load_params = dict()
        self.load_params["path"] = "."

        self.save_params = dict()
        self.save_params["base_path"] = "."
        self.save_params["save_path"] = None

        self.logger = logging.getLogger("QuadScanController.Controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("QuadScanController.__init__")

        self.state_lock = threading.Lock()      # Threading lock to avoid multithreading problems.
        # Use when risking concurrent access to objects.

        self.status = ""
        self.state = "unknown"
        self.state_notifier_list = list()       # Methods in this list will be called when the state
        # or status message is changed

        self.progress = 0                       # Progress indicator updated during time consuming tasks
        self.progress_notifier_list = list()

        self.load_pool = None
        self.save_pool = None
        self.process_pool = None            # type: TangoTwisted.ClearablePool
        self.process_count = 4

        if start is True:
            for key in self.device_names:
                self.device_factory_dict[key] = TangoAttributeFactory(self.device_names[key])
                self.device_factory_dict[key].startFactory()

    def read_attribute(self, name, device_name, use_tango_name=False):
        """
        Read a tango attribute from a device opened as a TangoAttributeFactory.

        :param name: Attribute name
        :param device_name: Device name to read from. This could be a key to the dict self.device_names
                            or be the tango device name xx/yy/zz
        :param use_tango_name: If True, use the direct tango device name. If False, use dict name
        :return: deferred that fires with the read attribute if successful
        """
        # self.logger.info("Read attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if use_tango_name is False:
            try:
                dev_name = self.device_names[device_name]
            except KeyError:
                er = ValueError("Device name {0} not found in {1}".format(device_name, self.device_names))
                self.logger.error(er)
                d = defer.Deferred()
                d.errback(er)
                return d

        else:
            dev_name = device_name
        try:
            factory = self.device_factory_dict[dev_name]
            d = factory.buildProtocol("read", name)
        except KeyError:
            er = ValueError("Device name {0} not found among {1}".format(dev_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def write_attribute(self, name, device_name, data, use_tango_name=False):
        self.logger.info("Write attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if use_tango_name is False:
            try:
                dev_name = self.device_names[device_name]
            except KeyError:
                er = ValueError("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
                self.logger.error(er)
                d = defer.Deferred()
                d.errback(er)
                return d

        else:
            dev_name = device_name
        try:
            factory = self.device_factory_dict[dev_name]
            d = factory.buildProtocol("write", name, data)
        except KeyError:
            er = ValueError("Device name {0} not found among {1}".format(dev_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def send_command(self, name, device_name, data, use_tango_name=False):
        self.logger.info("Send command \"{0}\" on \"{1}\"".format(name, device_name))
        if use_tango_name is False:
            try:
                dev_name = self.device_names[device_name]
            except KeyError:
                er = ValueError("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
                self.logger.error(er)
                d = defer.Deferred()
                d.errback(er)
                return d

        else:
            dev_name = device_name
        try:
            factory = self.device_factory_dict[dev_name]
            d = factory.buildProtocol("command", name, data)
        except KeyError:
            er = ValueError("Device name {0} not found among {1}".format(dev_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def add_device(self, dev_name):
        """
        Add a device to the device factory dict if not already there. After this it can be used to read and
        write attributes belonging to the device server.

        :param dev_name: Tango device name
        :return: Deferred that is fired when the factory is ready
        """
        self.logger.debug("Adding device {0}".format(dev_name))
        if dev_name not in self.device_factory_dict:
            fact = TangoAttributeFactory(dev_name)
            d = fact.startFactory()
            with self.state_lock:
                self.device_factory_dict[dev_name] = fact
            d.addErrback(self.add_device_error)
        else:
            self.logger.debug("Already in dict. Ignore.")
            d = defer.Deferred()
            d.callback(True)
        return d

    def add_device_error(self, err):
        """
        Callback for errors returned when creating device server factory.

        :param err: Error object
        :return:
        """
        self.logger.error("Error creating device: {0}".format(err))
        return err

    def defer_later(self, delay, delayed_callable, *a, **kw):
        """
        Call a function after a time interval, firing the returned deferred when completing.

        A deferred object is created with the delayed function set as the callback. Then a threading.Timer is
        created and started with the deferred.callback as the function to execute. Additional callbacks can then
        be added to the deferred, called when the delayed function completed.

        The deferred can be cancelled, cancelling the timer object.

        :param delay: Time to delay in s
        :param delayed_callable: Callable object
        :param a: Argument list
        :param kw: Keyword arg list
        :return: Deferred
        """
        self.logger.info("Calling {0} in {1} seconds".format(delayed_callable, delay))

        def defer_later_cancel(deferred):
            delayed_call.cancel()

        d = defer.Deferred(defer_later_cancel)
        d.addCallback(lambda ignored: delayed_callable(*a, **kw))
        delayed_call = threading.Timer(delay, d.callback, [None])
        delayed_call.start()
        return d

    def check_attribute(self, attr_name, device_name, target_value, period=0.3, timeout=1.0,
                        tolerance=None, write=True, use_tango_name=False):
        """
        Check an attribute to see if it reaches a target value. Returns a deferred for the result of the
        check.
        Upon calling the function the target is written to the attribute if the "write" parameter is True.
        Then reading the attribute is polled with the period "period" for a maximum number of retries.
        If the read value is within tolerance, the callback deferred is fired.
        If the read value is outside tolerance after retires attempts, the errback is fired.
        The maximum time to check is then period x retries

        :param attr_name: Tango name of the attribute to check, e.g. "fieldB"
        :param device_name: Device name to read from. This could be a key to the dict self.device_names
                         or be the tango device name xx/yy/zz
        :param target_value: Attribute value to wait for
        :param period: Polling period when checking the value
        :param timeout: Time to wait for the attribute to reach target value
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :param use_tango_name: If True, use the direct tango device name. If False, use dict name
        :return: Deferred that will fire depending on the result of the check
        """
        self.logger.info("Check attribute \"{0}\" on \"{1}\"".format(attr_name, device_name))
        if use_tango_name is False:
            try:
                dev_name = self.device_names[device_name]
            except KeyError:
                er = ValueError("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
                self.logger.error(er)
                d = defer.Deferred()
                d.errback(er)
                return d
        else:
            dev_name = device_name
        try:
            factory = self.device_factory_dict[dev_name]
            d = factory.buildProtocol("check", attr_name, None, write=write, target_value=target_value,
                                      tolerance=tolerance, period=period, timeout=timeout)
        except KeyError:
            er = ValueError("Device name {0} not found among {1}".format(dev_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def get_state(self):
        with self.state_lock:
            st = self.state
        return st

    def set_state(self, state, status=None):
        with self.state_lock:
            self.state = state
            if status is not None:
                self.status = status
        for m in self.state_notifier_list:
            m(state, status)
        self.state_change_signal.emit(state, status)

    def get_status(self):
        with self.state_lock:
            st = self.status
        return st

    def set_status(self, status_msg):
        self.logger.debug("Status: {0}".format(status_msg))
        with self.state_lock:
            self.status = status_msg
            state = self.state
        for m in self.state_notifier_list:
            m(state, status_msg)
        self.state_change_signal.emit(state, status_msg)

    def set_progress(self, progress):
        self.logger.debug("Setting progress to {0}".format(progress))
        with self.state_lock:
            self.progress = progress
        self.logger.debug("Notifying progress listeners")
        self.progress_signal.emit(progress)
        # for m in self.progress_notifier_list:
        #     m(progress)

    def get_progress(self):
        with self.state_lock:
            pr = self.progress
        return pr

    def add_state_notifier(self, state_notifier_method):
        self.state_notifier_list.append(state_notifier_method)

    def remove_state_notifier(self, state_notifier_method):
        try:
            self.state_notifier_list.remove(state_notifier_method)
        except ValueError:
            self.logger.warning("Method {0} not in list. Ignoring.".format(state_notifier_method))

    def add_progress_notifier(self, notifier_method):
        self.progress_notifier_list.append(notifier_method)

    def remove_progress_notifier(self, notifier_method):
        try:
            self.progress_notifier_list.remove(notifier_method)
        except ValueError:
            self.logger.warning("Method {0} not in list. Ignoring.".format(notifier_method))

    def set_parameter(self, state_name, param_name, value):
        self.logger.debug("Setting parameter {0}.{1} to {2}".format(state_name, param_name, value))
        with self.state_lock:
            if state_name == "load":
                self.load_params[param_name] = value
            elif state_name == "save":
                self.save_params[param_name] = value
            elif state_name == "analysis":
                self.analysis_params[param_name] = value
            elif state_name == "scan":
                self.scan_params[param_name] = value
            elif state_name == "idle":
                self.idle_params[param_name] = value
            elif state_name == "attr":
                self.attr_result[param_name] = value

    def get_parameter(self, state_name, param_name, dev_name=None):
        # self.logger.debug("Getting parameter {0}.{1}:".format(state_name, param_name))
        if dev_name is not None:
            key_name = dev_name + "/" + param_name
        else:
            key_name = param_name
        with self.state_lock:
            try:
                if state_name == "load":
                    value = self.load_params[key_name]
                elif state_name == "save":
                    value = self.save_params[key_name]
                elif state_name == "analysis":
                    value = self.analysis_params[key_name]
                elif state_name == "scan":
                    value = self.scan_params[key_name]
                elif state_name == "idle":
                    value = self.idle_params[key_name]
                elif state_name == "attr":
                    value = self.attr_result[key_name]
                else:
                    value = None
            except KeyError:
                self.logger.error("Get parameter key error: {0}, dict {1}".format(key_name, state_name))
                value = None
        # self.logger.debug("Value {0}".format(value))
        return value

    def set_result(self, state_name, result_name, value):
        self.logger.debug("Setting result {0}.{1}:".format(state_name, result_name))
        with self.state_lock:
            if state_name == "analysis":
                self.analysis_result[result_name] = value
            elif state_name == "scan":
                self.scan_result[result_name] = value

    def get_result(self, state_name, result_name):
        # self.logger.debug("Getting result {0}.{1}:".format(state_name, result_name))
        with self.state_lock:
            try:
                if state_name == "analysis":
                    value = self.analysis_result[result_name]
                elif state_name == "scan":
                    value = self.scan_result[result_name]
                else:
                    value = None
            except KeyError:
                value = None
        return value

    def update_attribute(self, result):
        try:
            attr_name = result.name
            # dev_name = result.device_name
        except AttributeError:
            return
        # self.logger.info("Updating result")
        # try:
        #     self.logger.debug("Result for {0}: {1}".format(result.name, result.value))
        # except AttributeError:
        #     return
        key_name = attr_name.lower()
        with self.state_lock:
            self.attr_result[key_name] = result
        self.attribute_ready_signal.emit(result)

    def get_attribute(self, attr_name, dev_name):
        """
        Return the stored device attribute from the dict attr_result

        :param attr_name: Tango name of attribute
        :param dev_name: Tango device name
        :return:
        """
        key_name = (dev_name + "/" + attr_name).lower()
        with self.state_lock:
            try:
                result = self.attr_result[key_name]
            except KeyError:
                result = None
        return result

    def init_scan_result_datastructure(self):
        """
        Clear old scan data and init with empty lists of lists with
        correct number of k_num elements.

        :return:
        """
        self.logger.info("Init scan result datastructure.")
        with self.state_lock:
            n_k = self.scan_params["num_k_values"]
            self.logger.info("Using {0} k values.".format(n_k))
            self.scan_result["raw_data"] = [list() for i in range(n_k)]
            self.scan_result["proc_data"] = [list() for i in range(n_k)]
            self.scan_result["line_data_x"] = [list() for i in range(n_k)]
            self.scan_result["line_data_y"] = [list() for i in range(n_k)]
            self.scan_result["x_cent"] = [list() for i in range(n_k)]
            self.scan_result["y_cent"] = [list() for i in range(n_k)]
            self.scan_result["sigma_x"] = [list() for i in range(n_k)]
            self.scan_result["sigma_y"] = [list() for i in range(n_k)]
            self.scan_result["k_data"] = [list() for i in range(n_k)]
            self.scan_result["enabled_data"] = [list() for i in range(n_k)]
            self.scan_result["charge_data"] = [list() for i in range(n_k)]

    def process_image(self, k_ind, image_ind):
        """
        Process an image in the stored scan raw_data. Put the process image back in the scan proc_data.

        Image processing consists of cropping, background subtracting, and filtering

        :param k_ind: index into the list k values of images
        :param image_ind: index into the list of images for each k value
        :return: processed image
        """
        self.logger.info("Processing image {0}, {1}".format(k_ind, image_ind))
        t0 = time.time()
        th = self.get_parameter("analysis", "threshold")
        roi_cent = self.get_parameter("scan", "roi_center")
        roi_dim = self.get_parameter("scan", "roi_dim")
        x = np.array([int(roi_cent[0] - roi_dim[0]/2.0), int(roi_cent[0] + roi_dim[0]/2.0)])
        y = np.array([int(roi_cent[1] - roi_dim[1]/2.0), int(roi_cent[1] + roi_dim[1]/2.0)])
        self.logger.debug("Threshold: {0}".format(th))
        self.logger.debug("ROI: {0}-{1}, {2}-{3}".format(x[0], x[1], y[0], y[1]))
        with self.state_lock:
            image_list = self.scan_result["raw_data"]
            # image_list = self.get_result("scan", "raw_data")
            # self.logger.debug("Image list size: {0} x {1}".format(len(image_list), len(image_list[0])))
            # self.logger.debug("Image type: {0}".format(image_list[0][0].dtype))
            try:
                pic = image_list[k_ind][image_ind]
            except IndexError as e:
                self.logger.warning("Could not get image {0}, {1}: {2}".format(k_ind, image_ind, e))
                return
            except TypeError as e:
                self.logger.warning("Could not get image {0}, {1}: {2}".format(k_ind, image_ind, e))
                return

            # Extract ROI and convert to double:
            pic_roi = np.double(pic[x[0]:x[1], y[0]:y[1]])

            # Normalize pic to 0-1 range, where 1 is saturation:
            if pic.dtype == np.int32:
                n = 2**16
            elif pic.dtype == np.uint8:
                n = 2**8
            else:
                n = 1

            # Median filtering:
            kernel = self.analysis_params["median_kernel"]
            pic_roi = medfilt2d(pic_roi / n, kernel)

            # Threshold image
            pic_roi[pic_roi < th] = 0.0

            line_x = pic_roi.sum(0)
            line_y = pic_roi.sum(1)
            q = line_x.sum()            # Total signal (charge) in the image

            proc_list = self.scan_result["proc_data"]
            line_data_x = self.scan_result["line_data_x"]
            line_data_y = self.scan_result["line_data_y"]

            cal = self.scan_params["pixel_size"]
            enabled = False
            l_x_n = np.sum(line_x)
            l_y_n = np.sum(line_y)
            # Enable point only if there is data:
            if l_x_n > 0.0:
                enabled = True
            x_v = cal[0] * np.arange(line_x.shape[0])
            y_v = cal[1] * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent)**2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent)**2 * line_y) / l_y_n)

            # Store processed data
            self.scan_result["enabled_data"][k_ind][image_ind] = enabled
            try:
                proc_list[k_ind][image_ind] = pic_roi
                line_data_x[k_ind][image_ind] = line_x
                line_data_y[k_ind][image_ind] = line_y
                self.scan_result["x_cent"][k_ind][image_ind] = x_cent
                self.scan_result["y_cent"][k_ind][image_ind] = y_cent
                self.scan_result["sigma_x"][k_ind][image_ind] = sigma_x
                self.scan_result["sigma_y"][k_ind][image_ind] = sigma_y
                self.scan_result["charge_data"][k_ind][image_ind] = q
            except IndexError:
                proc_list[k_ind].append(pic_roi)
                line_data_x[k_ind].append(pic_roi.sum(0))
                line_data_y[k_ind].append(pic_roi.sum(1))
                self.scan_result["x_cent"][k_ind].append(x_cent)
                self.scan_result["y_cent"][k_ind].append(y_cent)
                self.scan_result["sigma_x"][k_ind].append(sigma_x)
                self.scan_result["sigma_y"][k_ind].append(sigma_y)
                self.scan_result["charge_data"][k_ind].append(q)
        self.logger.debug("Image process time: {0}".format(time.time() - t0))

        self.image_done_signal.emit(k_ind, image_ind)
        return pic_roi

    def process_all_images(self):
        """
        Process all images in the scan. The processing is moved to threads to speed things up.

        :return: Deferred list that fires when all images are ready
        """
        d_list = []
        k_num = self.get_parameter("scan", "num_k_values")
        im_num = self.get_parameter("scan", "num_shots")
        total = np.double(k_num * im_num)

        for k_ind in range(k_num):
            for i_ind in range(im_num):
                d = TangoTwisted.defer_to_thread(self.process_image, k_ind, i_ind)
                d_list.append(d)
        dl = defer.DeferredList(d_list)
        dl.addCallbacks(self.process_images_done, self.process_image_error)
        return dl

    def process_all_images_pool(self):
        """
        Process all images in the scan. The processing is moved to a process pool to speed things up.
        This should be faster than using threads, unless the overhead of creating new process and
        moving data is larger than the actual computation...

        :return: Deferred list that fires when all images are ready
        """
        d_list = []
        k_num = self.get_parameter("scan", "num_k_values")
        im_num = self.get_parameter("scan", "num_shots")
        total = np.double(k_num * im_num)

        th = self.get_parameter("analysis", "threshold")
        roi_cent = self.get_parameter("scan", "roi_center")
        roi_dim = self.get_parameter("scan", "roi_dim")
        kernel = self.analysis_params["median_kernel"]
        cal = self.scan_params["pixel_size"]

        image_list = self.get_result("scan", "raw_data")

        if self.process_pool is None:
            self.process_pool = TangoTwisted.ClearablePool(processes=self.process_count)
        # else:
        #     self.process_pool.clear_pending_tasks()

        for k_ind in range(k_num):
            for i_ind in range(im_num):
                self.logger.debug("Starting process of {0}, {1}".format(k_ind, i_ind))
                try:
                    pic = image_list[k_ind][i_ind]
                except IndexError as e:
                    self.logger.warning("Could not get image {0}, {1}: {2}".format(k_ind, i_ind, e))
                    continue
                except TypeError as e:
                    self.logger.warning("Could not get image {0}, {1}: {2}".format(k_ind, i_ind, e))
                    continue
                # d = TangoTwisted.defer_to_pool(self.process_pool, self.process_image_pool, pic, k_ind, i_ind,
                #                                th, roi_cent, roi_dim, cal, kernel)
                # d = TangoTwisted.defer_to_pool(self.process_pool, test_image_pool, pic, k_ind, i_ind,
                #                                th, roi_cent, roi_dim, cal, kernel)
                d = self.process_pool.add_task(process_image_func,
                                               pic, k_ind, i_ind, th, roi_cent, roi_dim, cal, kernel)
                # d = self.process_pool.add_task(test_image_pool, k_ind, i_ind)

                d_list.append(d)
                d.addCallback(self.process_image_pool_store)
        dl = defer.DeferredList(d_list)
        dl.addCallbacks(self.process_images_done, self.process_image_error)
        return dl

    def process_image_pool(self, k_ind, image_ind):
        """
        Add a single image to the process_pool for roi extraction, thresholding, and filtering.
        After completing the image is sent to process_image_pool_store for storing the processed
        image. This is because the process has to be done with on a non-class method, and so
        does not know the controller object.

        :param k_ind: Index into the list of k-values taken during the scan
        :param image_ind: Indoex into the list of images taken for this k-value
        :return: Deferred that fires when the image processing is completed
        """
        self.logger.info("Processing image {0}, {1} in pool".format(k_ind, image_ind))
        t0 = time.time()
        if self.process_pool is None:
            self.process_pool = TangoTwisted.ClearablePool(processes=self.process_count)

        th = self.get_parameter("analysis", "threshold")
        roi_cent = self.get_parameter("scan", "roi_center")
        roi_dim = self.get_parameter("scan", "roi_dim")
        kernel = self.analysis_params["median_kernel"]
        cal = self.get_parameter("scan", "pixel_size")
        bpp = self.get_parameter("scan", "bpp")

        image_list = self.get_result("scan", "raw_data")

        # Init pool if there is none
        if self.process_pool is None:
            self.process_pool = TangoTwisted.ClearablePool(processes=self.process_count)

        pic = image_list[k_ind][image_ind]
        d = self.process_pool.add_task(process_image_func,
                                       pic, k_ind, image_ind, th, roi_cent, roi_dim, cal, kernel, bpp)
        d.addCallback(self.process_image_pool_store)

        return d

    def process_image_pool_store(self, result):
        """
        Stores a processed image. All parameters to identify the image is in the result data structure.
        The image_done_signal is emitted at the end of this method.

        :param result: Proessed image deferred result
        :return: Same result for further deferred callback processing
        """
        self.logger.info("Storing processed image")
        k_ind = result["k_ind"]
        image_ind = result["image_ind"]
        pic_roi = result["pic_roi"]
        try:
            self.logger.info("Storing processed image {0}, {1}. Size {2}".format(k_ind, image_ind, pic_roi.shape))
        except AttributeError as e:
            self.logger.info("Pic_roi attribute error: {0}".format(e))
        line_x = result["line_x"]
        line_y = result["line_y"]
        x_cent = result["x_cent"]
        y_cent = result["y_cent"]
        sigma_x = result["sigma_x"]
        sigma_y = result["sigma_y"]
        q = result["q"]
        enabled = result["enabled"]

        proc_list = self.scan_result["proc_data"]
        line_data_x = self.scan_result["line_data_x"]
        line_data_y = self.scan_result["line_data_y"]
        old_enabled = self.scan_result["enabled_data"][k_ind][image_ind]
        self.logger.debug("Old enabled: {0}".format(old_enabled))
        self.scan_result["enabled_data"][k_ind][image_ind] = enabled
        try:
            proc_list[k_ind][image_ind] = pic_roi
            line_data_x[k_ind][image_ind] = line_x
            line_data_y[k_ind][image_ind] = line_y
            self.scan_result["x_cent"][k_ind][image_ind] = x_cent
            self.scan_result["y_cent"][k_ind][image_ind] = y_cent
            self.scan_result["sigma_x"][k_ind][image_ind] = sigma_x
            self.scan_result["sigma_y"][k_ind][image_ind] = sigma_y
            self.scan_result["charge_data"][k_ind][image_ind] = q
        except IndexError:
            self.logger.debug("Appending value")
            proc_list[k_ind].append(pic_roi)
            line_data_x[k_ind].append(pic_roi.sum(0))
            line_data_y[k_ind].append(pic_roi.sum(1))
            self.logger.debug("Value appended")
            self.scan_result["x_cent"][k_ind].append(x_cent)
            self.scan_result["y_cent"][k_ind].append(y_cent)
            self.logger.debug("Centroids")
            self.scan_result["sigma_x"][k_ind].append(sigma_x)
            self.scan_result["sigma_y"][k_ind].append(sigma_y)
            self.logger.debug("Sigma")
            self.scan_result["charge_data"][k_ind].append(q)
        self.image_done_signal.emit(k_ind, image_ind)
        self.logger.debug("Signal emitted")
        return result

    def process_images_done(self, result):
        self.logger.info("All images processed.")
        self.logger.info("Fitting image data")
        self.processing_done_signal.emit()
        # self.process_pool.close()
        # self.process_pool = None
        self.fit_quad_data()

    def process_image_error(self, error):
        self.logger.error("Process image error: {0}".format(error))

    def fit_quad_data(self):
        algo = self.get_parameter("analysis", "fit_algo")
        self.logger.debug("Fitting data with {0} algorithm".format(algo))
        if algo == "full_matrix":
            self.fit_full_transfer_matrix()
        elif algo == "thin_lens":
            self.fit_thin_lens()
        else:
            self.logger.warning("Fitting algorithm {0} not found.".format(algo))
            return False
        self.fit_done_signal.emit()
        return True

    def fit_thin_lens(self):
        self.logger.info("Fitting image data using thin lens approximation")
        k_data = np.array(self.get_result("scan", "k_data")).flatten()
        sigma_data = np.array(self.get_result("scan", "sigma_x")).flatten()
        en_data = np.array(self.get_result("scan", "enabled_data")).flatten()
        try:
            s2 = (sigma_data[en_data]) ** 2
        except IndexError as e:
            self.logger.warning("Could not address enabled sigma values. "
                                "En_data: {0}, sigma_data: {1}".format(en_data, sigma_data))
            return
        k = k_data[en_data]
        ind = np.isfinite(s2)
        poly = np.polyfit(k[ind], s2[ind], 2)
        self.set_result("scan", "fit_poly", poly)
        self.logger.debug("Fit coefficients: {0}".format(poly))
        d = self.get_parameter("analysis", "quad_screen_dist")
        L = self.get_parameter("analysis", "quad_length")
        gamma = self.get_parameter("analysis", "electron_energy") / 0.511
        self.logger.debug("d: {0}, poly[0]: {1}, poly[1]: {2}, poly[2]: {3}".format(d, poly[0], poly[1], poly[2]))
        eps = 1 / (d ** 2 * L) * np.sqrt(poly[0] * poly[2] - poly[1] ** 2 / 4)
        eps_n = eps * gamma
        beta = poly[0] / (eps * d ** 2 * L ** 2)
        alpha = (beta + poly[1] / (2 * eps * d * L)) / L
        self.logger.info("-------------------------------")
        self.logger.info("eps_n  = {0:.3f} mm x mrad".format(eps_n * 1e6))
        self.logger.info("beta   = {0:.4g} m".format(beta))
        self.logger.info("alpha  = {0:.4g} rad".format(alpha))
        self.logger.info("-------------------------------")
        self.set_result("analysis", "eps", eps_n)
        self.set_result("analysis", "beta", beta)
        self.set_result("analysis", "alpha", alpha)

        x = np.linspace(k.min(), k.max(), 100)
        y = np.polyval(poly, x)
        self.set_result("analysis", "fit_data", [x, np.sqrt(y)])

    def fit_full_transfer_matrix(self):
        self.logger.info("Fitting using full transfer matrix")
        k_data = np.array(self.get_result("scan", "k_data")).flatten()
        sigma_data = np.array(self.get_result("scan", "sigma_x")).flatten()
        en_data = np.array(self.get_result("scan", "enabled_data")).flatten()
        d = self.get_parameter("analysis", "quad_screen_dist")
        L = self.get_parameter("analysis", "quad_length")
        gamma_energy = self.get_parameter("analysis", "electron_energy") / 0.511
        self.logger.debug("sigma_data: {0}".format(sigma_data.shape))
        self.logger.debug("en_data: {0}".format(en_data.shape))
        self.logger.debug("k_data: {0}".format(k_data))
        try:
            s2 = (sigma_data[en_data]) ** 2
        except IndexError as e:
            self.logger.warning("Could not address enabled sigma values. "
                                "En_data: {0}, sigma_data: {1}".format(en_data, sigma_data))
            return

        k = k_data[en_data]
        ind = np.isfinite(s2)

        k_sqrt = np.sqrt(k[ind]*(1+0j))
        self.logger.debug("k_sqrt = {0}".format(k_sqrt))
        A = np.real(np.cos(k_sqrt * L) - d * k_sqrt * np.sin(k_sqrt * L))
        B = np.real(1 / k_sqrt * np.sin(k_sqrt * L) + d * np.cos(k_sqrt * L))
        M = np.vstack((A*A, -2*A*B, B*B)).transpose()
        try:
            x = np.linalg.lstsq(M, s2[ind])
        except Exception as e:
            self.logger.error("Error when fitting lstsqr: {0}".format(e))
            return
        self.logger.debug("Fit coefficients: {0}".format(x[0]))
        eps = np.sqrt(x[0][2] * x[0][0] - x[0][1]**2)
        eps_n = eps * gamma_energy
        beta = x[0][0] / eps
        alpha = x[0][1] / eps

        self.logger.info("-------------------------------")
        self.logger.info("eps_n  = {0:.3f} mm x mrad".format(eps_n * 1e6))
        self.logger.info("beta   = {0:.4g} m".format(beta))
        self.logger.info("alpha  = {0:.4g} rad".format(alpha))
        self.logger.info("-------------------------------")
        self.set_result("analysis", "eps", eps_n)
        self.set_result("analysis", "beta", beta)
        self.set_result("analysis", "alpha", alpha)

        x = np.linspace(k.min(), k.max(), 100)
        x_sqrt = np.sqrt(x*(1+0j))
        Ax = np.real(np.cos(x_sqrt * L) - d * x_sqrt * np.sin(x_sqrt * L))
        Bx = np.real(1 / x_sqrt * np.sin(x_sqrt * L) + d * np.cos(x_sqrt * L))

        y = Ax**2 * beta * eps - 2 * Ax * Bx * alpha * eps + Bx**2 * (1 + alpha**2) * eps / beta
        self.set_result("analysis", "fit_data", [x, np.sqrt(y)])

    def set_analysis_parameters(self, quad_length, quad_screen_distance, electron_energy):
        self.logger.info("Setting analysis accelerator parameters: L={0}, d={1}, E={2}".format(quad_length,
                                                                                               quad_screen_distance,
                                                                                               electron_energy))
        self.set_parameter("analysis", "quad_length", quad_length)
        self.set_parameter("analysis", "quad_screen_dist", quad_screen_distance)
        self.set_parameter("analysis", "electron_energy", electron_energy)

    def add_raw_image(self, k_num, k_value, image):
        self.logger.info("Adding new image with k index {0}".format(k_num))
        with self.state_lock:
            im_list = self.scan_result["raw_data"]
            k_list = self.scan_result["k_data"]
            en_list = self.scan_result["enabled_data"]
            try:
                im_list[int(k_num)].append(image)
                k_list[int(k_num)].append(k_value)
                en_list[int(k_num)].append(False)
            except IndexError:
                self.logger.warning("Image index out of range: {0}/{1}, skipping".format(k_num,
                                                                                         len(self.scan_raw_data)))

    def populate_matching_sections(self):
        """
        Populate matching section data by quering tango database properties.
        Called when entering Database state.

        This does not establish a connection with the device servers. That is done in
        set_section method.

        Data retrieved:
        Quads... name, length, position, polarity
        Screens... name, position
        :return:
        """
        self.logger.info("Populating matching sections by checking tango database")
        db = tango.Database()
        self.logger.debug("Database retrieved")

        sections = self.get_parameter("scan", "sections")
        sect_quads = self.get_parameter("scan", "section_quads")
        sect_screens = self.get_parameter("scan", "section_screens")

        # Loop through sections to find matching devices based on their names:
        for s in sections:
            # Quad names are e.g. i-ms1/mag/qb-01
            quad_dev_list = db.get_device_exported("*{0}*/mag/q*".format(s)).value_string
            quad_list = list()
            for q in quad_dev_list:
                quad = dict()
                try:
                    # Extract data for each found quad:
                    quad["name"] = q.split("/")[-1].lower()
                    p = db.get_device_property(q, ["__si", "length", "polarity", "circuitproxies"])
                    quad["position"] = np.double(p["__si"][0])
                    quad["length"] = np.double(p["length"][0])
                    quad["polarity"] = np.double(p["polarity"][0])
                    quad["crq"] = p["circuitproxies"][0]

                    quad_list.append(quad)
                except IndexError as e:
                    self.logger.error("Index error when parsing quad {0}: {1}".format(q, e))
                except KeyError as e:
                    self.logger.error("Key error when parsing quad {0}: {1}".format(q, e))

            # Screen names are e.g. i-ms1/dia/scrn-01
            screen_dev_list = db.get_device_exported("*{0}*/dia/scrn*".format(s)).value_string
            screen_list = list()
            for sc in screen_dev_list:
                scr = dict()
                try:
                    # Extract data for each found screen
                    scr["name"] = sc.split("/")[-1].lower()
                    scr["position"] = np.double(db.get_device_property(sc, "__si")["__si"][0])
                    screen_list.append(scr)
                except IndexError as e:
                    self.logger.error("Index error when parsing screen {0}: {1}".format(sc, e))
                except KeyError as e:
                    self.logger.error("Key error when parsing screen {0}: {1}".format(sc, e))

            sect_quads[s] = quad_list
            sect_screens[s] = screen_list
            self.logger.debug("Populating section {0}:".format(s.upper()))
            self.logger.debug("Found quads: {0}".format(quad_list))
            self.logger.debug("Found screens: {0}".format(screen_list))

    def set_section(self, sect_name, quad_name, screen_name):
        """
        Set the current section, quad, and screen. Update the name of current sect, quad , scrn in
        scan_params dict. If a new setting is selected: load aux devices into quad_device_names and
        screen_device_names. Update quad and screen parameters (pos, length, ...)

        It assumes that populate_matching_sections is completed for information about the different
        devices in the section.

        :param sect_name:
        :param quad_name:
        :param screen_name:
        :return:
        """
        self.logger.info("Setting section {0}, quad {1}, screen {2}".format(sect_name, quad_name, screen_name))
        snl = sect_name.lower()
        sn = None
        current_sect = self.get_parameter("scan", "section_name")
        update_all = False
        if "ms1" in snl:
            sn = "ms1"
            if sn != current_sect:
                self.set_parameter("scan", "section_name", "ms1")
                update_all = True
        elif "ms2" in snl:
            sn = "ms2"
            if sn != current_sect:
                self.set_parameter("scan", "section_name", "ms2")
                update_all = True
        elif "ms3" in snl:
            sn = "ms3"
            if sn != current_sect:
                self.set_parameter("scan", "section_name", "ms3")
                update_all = True
        elif "sp02" in snl:
            sn = "sp02"
            if sn != current_sect:
                self.set_parameter("scan", "section_name", "sp02")
                update_all = True

        # TODO: Make sure the quad_name and screen_name are in self.get_parameter("scan", "section_quads") and
        # TODO: self.get_parameter("scan", "section_screens")

        dl = list()

        # Check if we need to update quad selection
        if update_all is True or self.get_parameter("scan", "quad_name") != quad_name:
            update_quad = True
        else:
            update_quad = False

        q_pos = None
        if update_quad is True:
            # We need to control the magnet and the corresponding crq device (which is where the actual
            # k-value is set)
            quad_devices = dict()
            quad_num = quad_name.split("-")[-1]
            quad_mag_name = "i-{0}/mag/{1}".format(sn, quad_name)       # k-value
            quad_crq_name = "i-{0}/mag/crq-{1}".format(sn, quad_num)    # electron energy, k-value
            quad_devices["mag"] = quad_mag_name
            quad_devices["crq"] = quad_crq_name
            for dev_name in quad_devices.itervalues():
                dl.append(self.add_device(dev_name))

            try:
                quad_list = self.get_parameter("scan", "section_quads")[sn]
                # The "section_quads" is populated in populate_matching_sections
            except KeyError:
                self.logger.error("Section {0} not found in section_quads dict".format(sn))
                return
            L = None
            d = None
            for qd in quad_list:
                if qd["name"] == quad_name:
                    L = qd["length"]
                    q_pos = qd["position"]
            self.set_parameter("scan", "quad_length", L)
            self.logger.debug("Connecting to quad devices {0}".format(quad_devices))
            self.set_parameter("scan", "quad_device_names", quad_devices)
            self.set_parameter("scan", "quad_name", quad_name)
            self.set_parameter("scan", "quad_pos", q_pos)

        # Check if we need to update screen selection
        if update_all is True or self.get_parameter("scan", "screen_name") != screen_name:
            update_screen = True
        else:
            update_screen = False

        s_pos = None
        if update_screen is True:
            try:
                screen_list = self.get_parameter("scan", "section_screens")[sn]
            except KeyError:
                self.logger.error("Section {0} not found in section_screens dict".format(sn))
                return

            for sc in screen_list:
                if sc["name"] == screen_name:
                    s_pos = sc["position"]

            scrn_devices = dict()
            screen_dev_name = "i-{0}/dia/{1}".format(sn, screen_name)
            cam_name = "i-{0}-dia-{1}".format(sn, screen_name)
            camera_ctrl_name = "lima/limaccd/{0}".format(cam_name)          # Controlling camera exposure, gain
            camera_view_name = "lima/liveviewer/{0}".format(cam_name)       # Get image, start capturing
            camera_beam_name = "lima/beamviewer/{0}".format(cam_name)       # ROI
            scrn_devices["ctrl"] = camera_ctrl_name
            scrn_devices["view"] = camera_view_name
            scrn_devices["beam"] = camera_beam_name
            scrn_devices["screen"] = screen_dev_name
            self.logger.debug("Connecting to screen devices {0}".format(scrn_devices))
            for dev_name in scrn_devices.itervalues():
                dl.append(self.add_device(dev_name))
            self.set_parameter("scan", "screen_device_names", scrn_devices)
            self.set_parameter("scan", "screen_name", screen_name)
            self.set_parameter("scan", "screen_pos", s_pos)

        if s_pos is not None or q_pos is not None:
            qp = self.get_parameter("scan", "quad_pos")
            sp = self.get_parameter("scan", "screen_pos")
            d = sp - qp
            self.set_parameter("scan", "quad_screen_distance", d)

        def_list = defer.DeferredList(dl)
        if update_screen is True:
            # Read the roi stored in the camera device if a new screen was selected
            def_list.addCallback(self.read_camera_parameters)
        if update_quad is True:
            def_list.addCallback(self.read_electron_energy)
        return def_list

    def read_camera_parameters(self, result):
        """
        Callback to read camera ROI from beamviewer device.
        Creates a deferred with update_camera_roi as callback

        :param result: Deferred result (not used)
        :return: New deferred that fires when the read is finished
        """
        self.logger.debug("Reading camera ROI")
        beamviewer_name = self.get_parameter("scan", "screen_device_names")["beam"]
        self.logger.debug("Beamviewer name: {0}".format(beamviewer_name))
        d = self.read_attribute("roi", beamviewer_name, use_tango_name=True)
        d.addCallback(self.update_camera_roi)

        ctrl_name = self.get_parameter("scan", "screen_device_names")["ctrl"]
        d_bpp = self.read_attribute("image_type", ctrl_name, use_tango_name=True)
        d_bpp.addCallback(self.update_camera_param)
        d_pix = self.read_attribute("camera_pixelsize", ctrl_name, use_tango_name=True)
        d_pix.addCallback(self.update_camera_param)
        return d

    def update_camera_roi(self, result):
        """
        Callback to update the camera roi stored in the scan params dict.

        :param result: Deferred result containing the device roi attribute
        :return:
        """
        self.logger.info("Updating camera ROI")
        try:
            # The returned ROI is a string with the roi data as elements in a list
            # "[min_x, max_x, min_y, max_y]"
            # E.g. "[0, 1280, 0, 1024]"
            raw_data = result.value
            roi_data = np.array(raw_data.strip("[]").split(",")).astype(np.int)
        except AttributeError:
            return False
        roi_center = ((roi_data[1] + roi_data[0]) / 2.0, (roi_data[2] + roi_data[3]) / 2.0)
        roi_dim = ((roi_data[1] - roi_data[0]), (roi_data[3] - roi_data[2]))
        self.set_parameter("scan", "roi_center", roi_center)
        self.set_parameter("scan", "roi_dim", roi_dim)
        self.camera_roi_read_done_signal.emit()
        return result

    def update_camera_param(self, result):
        try:
            param_name = result.name.lower()
        except AttributeError as e:
            self.logger.error("Error updating camera parameter: {0}".format(e))
        self.logger.info("Updating parameter {0} for selected camera to {1}".format(param_name, result.value))
        if param_name == "image_type":
            s = result.value.lower()
            bpp = int(s.split("bpp")[1])
            self.set_parameter("scan", "bpp", bpp)
        elif param_name == "camera_pixelsize":
            cal = result.value
            self.set_parameter("scan", "pixel_size", cal)

    def read_electron_energy(self, result):
        """
        Callback to read electron energy from crq device.
        Creates a deferred with update_electron_energy as callback

        :param result: Deferred result (not used)
        :return: New deferred that fires when the read is finished
        """
        self.logger.debug("Reading electron energy")
        crq_name = self.get_parameter("scan", "quad_device_names")["crq"]
        self.logger.debug("Crq name: {0}".format(crq_name))
        d = self.read_attribute(["energy", "mainfieldcomponent"], crq_name, use_tango_name=True)

        d.addCallback(self.update_electron_energy)
        return d

    def update_electron_energy(self, result):
        """
        Callback to update the electron energy stored in the scan params dict.

        :param result: Deferred result containing the device energy attribute
        :return:
        """
        try:
            energy = result[0].value * 1e-6        # Value from tango device in eV, we want MeV
            k = result[1].value
            self.logger.info("Updating electron energy to {0}".format(energy))
            self.logger.info("Updating mainfieldcomponent to {0}".format(k))
        except AttributeError:
            self.logger.error("AttributeError Updating electron energy")
            return False
        self.set_parameter("scan", "electron_energy", energy)
        # self.update_attribute(result[1])
        self.set_parameter("attr", "mainfieldcomponent", result[1])
        self.quad_selected_signal.emit()
        return result

    def exit(self):
        self.logger.info("Exiting controller. Process pools stopping.")
        if self.process_pool is not None:
            try:
                self.process_pool.stop()
            except AttributeError:
                pass
        if self.load_pool is not None:
            try:
                self.load_pool.stop()
            except AttributeError:
                pass
        if self.save_pool is not None:
            try:
                self.save_pool.stop()
            except AttributeError:
                pass


class Scan(object):
    """
    Run a scan of one attribute while measuring another
    """
    def __init__(self, controller, scan_attr_name, scan_dev_name, start_pos, stop_pos, step,
                 meas_attr_name, meas_dev_name, averages=1, meas_callable=None, meas_new_id=None):
        """

        :param controller: QuadScanController object
        :param scan_attr_name: Name of attribute to scan (e.g. mainfieldcomponent)
        :param scan_dev_name: Tango name of device that has the scan attribute (e.g. i-ms1/mag/crq-01)
        :param start_pos: Start value of the scan attribute
        :param stop_pos: End value of the scan attribute
        :param step: Value change of the scan attribute for each step
        :param meas_attr_name: Name of attribute to measure at each step of scan (e.g image)
        :param meas_dev_name: Tango name of device that has the measurement attribute
        :param averages: Number of readings of the measurement attribute at each scan point to average
        :param meas_callable: Additional callable that is called at end of each scan point to save data etc...
        :param meas_new_id: Additional attribute in measurement device to check if new data
                            is available (e.g. imagecounter)
        """
        self.controller = controller    # type: QuadScanController.QuadScanController
        self.scan_attr = scan_attr_name
        self.scan_dev = scan_dev_name
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.step = step
        self.meas_attr = meas_attr_name
        self.meas_dev = meas_dev_name

        self.use_tango_name = True

        self.current_pos = None
        self.current_meas_ind = 0
        self.current_pos_ind = 0
        self.averages = averages
        self.pos_data = []
        self.meas_data = []
        self.scan_start_time = None
        self.scan_arrive_time = time.time()
        self.data_time = time.time()
        self.status_update_time = time.time()
        self.status_update_interval = 1.0           # Status messages are sent with this interval to the controller

        self.meas_callable = meas_callable          # Call this function when taking a new step in the scan
        self.meas_new_id = meas_new_id              # Attribute name to read to check if there is new data
        self.meas_max_time = 1.5                    # Max time to wait for new data
        self.meas_start_time = None                 # Holds the starting time for checking new data
        self.meas_last_hash = 0.0                   # Holds the last hash value to see if it changed

        self.simulation = False                     # If simulation is True, don't actually change the scan attribute

        self.logger = logging.getLogger("QuadScanController.Scan_{0}_{1}".format(self.scan_attr, self.meas_attr))
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()
        self.cancel_flag = False

        # Add errback handling!

    def start_scan(self):
        self.logger.info("Starting scan of {0} from {1} to {2} measuring {3}".format(self.scan_attr.upper(),
                                                                                     self.start_pos,
                                                                                     self.stop_pos,
                                                                                     self.meas_attr.upper()))
        self.scan_start_time = time.time()
        self.data_time = time.time()
        self.status_update_time = self.scan_start_time
        self.cancel_flag = False
        scan_pos = self.start_pos
        tol = self.step * 0.1

        d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos,
                                             0.1, 10.0, tolerance=tol, write=True, use_tango_name=self.use_tango_name)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        # d0.addCallback(lambda _: self.controller.read_attribute(self.meas_attr, self.meas_dev))
        # d0.addCallback(self.meas_scan)
        self.d = defer.Deferred(self.cancel_scan)

        return self.d

    def simulate_scan(self):
        self.simulation = True
        self.logger.info("Starting scan simulation of {0} from {1} to {2} measuring {3}".format(self.scan_attr.upper(),
                                                                                                self.start_pos,
                                                                                                self.stop_pos,
                                                                                                self.meas_attr.upper()))
        self.scan_start_time = time.time()
        self.data_time = time.time()
        self.status_update_time = self.scan_start_time
        self.cancel_flag = False
        scan_pos = self.start_pos
        tol = self.step * 0.1

        d0 = self.controller.read_attribute(self.scan_attr, self.scan_dev, use_tango_name=self.use_tango_name)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        # d0.addCallback(lambda _: self.controller.read_attribute(self.meas_attr, self.meas_dev))
        # d0.addCallback(self.meas_scan)
        self.d = defer.Deferred(self.cancel_scan)

        return self.d

    def scan_step(self):
        """
        Check if it is time to end the scan. If not, issue a move to the next scan position.
        Callback scan_arrive is called when the move is done.
        :return:
        """
        self.current_pos_ind += 1
        self.logger.info("Scan step {0}".format(self.current_pos_ind))
        # Progress level:
        p = (self.current_pos - self.start_pos) / (self.stop_pos-self.start_pos)
        self.controller.progress_signal.emit(p)
        tol = self.step * 0.1   # Tolerance for scan attribute
        scan_pos = self.current_pos + self.step     # New scan position
        if scan_pos > self.stop_pos or self.cancel_flag is True:
            self.scan_done()
            return self.d

        if self.simulation is True:
            d0 = self.controller.read_attribute(self.scan_attr, self.scan_dev, use_tango_name=self.use_tango_name)
        else:
            d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos, 0.1, 3.0, tolerance=tol,
                                                 write=True, use_tango_name=self.use_tango_name)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        return d0

    def scan_arrive(self, result):
        """
        Arrived at the new scan position. Wait a time interval corresponding to the time between shots
        and then issue a read of measurement attribute. Callback meas_read is called with the measured
        value.

        :param result: Deferred result. Used to get current position.
        :return:
        """
        if self.simulation is True:
            # If simulating, add the step anyway to be able to stop the scan
            if self.current_pos is None:
                self.current_pos = 0.0
            self.current_pos += self.step
        else:
            try:
                self.current_pos = result.value
            except AttributeError:
                self.logger.error("Scan arrive not returning attribute: {0}".format(result))
                return False

        self.logger.info("Scan arrive at pos {0}".format(self.current_pos))
        t = time.time()
        # If the time since last update is larger than the update interval,
        # set a new status message
        if t - self.status_update_time > self.status_update_interval:
            status = "Scanning from {0} to {1} with step size {2}\n" \
                     "Position: {3}".format(self.start_pos, self.stop_pos, self.step, self.current_pos)
            self.controller.set_status(status)
            self.status_update_time = t

        # After arriving, check the time against the last read data. Wait for a time so that the next
        # data frame should be there.
        self.scan_arrive_time = t
        try:
            wait_time = (self.scan_arrive_time - self.data_time) % (1.0 / self.controller.idle_params["reprate"])
        except KeyError:
            # If the reprate is not specified, assume 10Hz
            wait_time = 0.1
        d0 = defer_later(wait_time, self.meas_issue_new_read)
        d0.addErrback(self.scan_error_cb)
        self.logger.debug("Scheduling read in {0} s".format(wait_time))

        return True

    def meas_issue_new_read(self):
        """
        Called after scan_arrive to read a new data point.
        :return:
        """
        self.logger.info("Reading measurement")
        self.meas_start_time = time.time()
        if self.meas_new_id is not None:
            # The meas_new_id was supplied, so try checking for a change in hash value.
            d0 = self.controller.read_attribute(self.meas_new_id, self.meas_dev, use_tango_name=self.use_tango_name)
            d0.addCallbacks(self.meas_check_hash, self.scan_error_cb)
        else:
            # No meas_new_id, just read the attribute directly
            d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev, use_tango_name=self.use_tango_name)
            d0.addCallbacks(self.meas_scan_store, self.scan_error_cb)
        return True

    def meas_check_hash(self, result):
        """
        Callback from read_attribute for checking if new measurement is available by calculating a "hash value"
        (i.e. summing the values in the attribute data)
        :param result: Deferred callback result
        :return:
        """
        hash_val = np.sum(result.value)
        if (hash_val == self.meas_last_hash) and (time.time() - self.meas_start_time > self.meas_max_time):
            # The hash value did not change and the total time elapsed is smaller than the acceptable (meas_max_time).
            # So there is no new measurement data. Try again.
            d0 = self.controller.read_attribute(self.meas_new_id, self.meas_dev, use_tango_name=self.use_tango_name)
            d0.addCallbacks(self.meas_check_hash, self.scan_error_cb)
        else:
            # The hash value change or the time is too long waiting. Read the actual measurement data.
            self.meas_last_hash = hash_val
            d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev, use_tango_name=self.use_tango_name)
            d0.addCallbacks(self.meas_scan_store, self.scan_error_cb)
        return result

    def meas_scan_store(self, result):
        """
        Called when the new measurement is done. Stores the data internally
        :param result: Result from the read_attribute deferred
        :return:
        """
        measure_value = result.value
        self.logger.debug("Measure at scan pos {0}, pos ind {1}, "
                          "meas ind {2}, result: {3}".format(self.current_pos, self.current_pos_ind,
                                                             self.current_meas_ind, measure_value))

        # This is a good idea, but the timestamp only reflects the time of the read_attribtue, not
        # the actual time the measurement was taken. Also the machines are not synchronized so
        # timestamps from different computers can't be compared.
        # First check if this was taken before the scan arrived to the new position, then re-read
        # if result.time.totime() <= self.scan_arrive_time:
        #     self.logger.debug("Old data. Wait for new.")
        #     t = time.time() - result.time.totime()
        #     if t > 2.0:
        #         self.logger.error("Timeout waiting for new data. {0} s elapsed".format(t))
        #         self.scan_error_cb(RuntimeError("Timeout waiting for new data"))
        #         return False
        #     d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev, use_tango_name=self.use_tango_name)
        #     d0.addCallbacks(self.meas_scan_store, self.scan_error_cb)
        #     return False
        self.data_time = result.time.totime()

        self.meas_data.append(measure_value)
        self.pos_data.append(self.current_pos)
        if self.meas_callable is not None:
            self.meas_callable(self.current_pos_ind, self.current_meas_ind, self.current_pos, result)
        self.current_meas_ind += 1
        if self.current_meas_ind < self.averages or self.cancel_flag is True:
            t = time.time()
            try:
                wait_time = (t - self.data_time) % (1.0 / self.controller.idle_params["reprate"])
            except KeyError:
                # No reprate param, default to 0.1 s wait time
                wait_time = 0.1
            d0 = defer_later(wait_time, self.meas_issue_new_read)
            d0.addErrback(self.scan_error_cb)
        else:
            self.current_meas_ind = 0
            self.scan_step()    # Do a new step
        return True

    def scan_done(self):
        self.logger.info("Scan done!")
        scan_raw_data = np.array(self.meas_data)
        self.logger.info("Scan dimensions: {0}".format(scan_raw_data.shape))
        pos_data = np.array(self.pos_data)
        self.d.callback([pos_data, scan_raw_data, self.scan_start_time])

    def scan_error_cb(self, err):
        self.logger.error("Scan error: {0}".format(err))
        # Here we can handle the error if it is manageable or
        # propagate the error to the calling callback chain:
        if err.type in []:
            pass
        elif err.type == tango.DevFailed:
            err_t = err[0]
            if err_t.reason == "API_WAttrOutsideLimit":
                self.controller.set_status("Scan parameter outside limit")
                self.d.errback(err)
        else:
            self.d.errback(err)

    def cancel_scan(self, result):
        self.logger.info("Cancelling scan, result {0}".format(result))
        self.cancel_flag = True


class QuadScanAnalyse(object):
    """
    Steps to take during analysis:

    1. Load data
    2. Pre-process data... threshold, background subtract, filter
    3. Calculate beam size
    4. Fit 2:nd order polynomial to data

    """
    def __init__(self, controller):

        self.controller = controller    # type: QuadScanController.QuadScanController

        self.scan_raw_data = None
        self.scan_proc_data = None
        self.scan_roi_data = None
        self.magnet_data = None

        self.logger = logging.getLogger("QuadScanController.Analysis")
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()

    def start_analysis(self):
        self.logger.info("Starting up quadscan analysis")
        self.load_data()
        self.preprocess()
        self.find_roi()
        self.convert_data(use_roi=True)
        d = TangoTwisted.defer_to_thread(self.fit_scan)
        d.addCallbacks(self.retrieve_data, self.analysis_error)
        self.d = defer.Deferred()
        return self.d

    def load_data(self):
        self.logger.info("Loading data from scan")
        scan_result = self.controller.scan_result
        pos = np.array(scan_result["pos_data"])  # Vector containing the motor positions during the scan
        # The wavelengths should have been read earlier.
        self.magnet_data = pos
        self.scan_raw_data = np.array(scan_result["scan_data"])
        if self.time_data.shape[0] != self.scan_raw_data.shape[0]:
            err_s = "Time vector not matching scan_data dimension: {0} vs {1}".format(self.time_data.shape[0],
                                                                                      self.scan_raw_data.shape[0])
            self.logger.error(err_s)
            fail = Failure(AttributeError(err_s))
            self.d.errback(fail)
            return

    def preprocess(self):
        """
        Preprocess data to improve the quad scan fit quality.
        The most important part is the thresholding to isolate the data part
        of the image.
        We also do background subtraction, normalization, and filtering.
        Thresholding is done after background subtraction.
        :return:
        """
        self.logger.info("Preprocessing scan data")
        if self.controller.analyse_params["background_subtract"] is True:
            # Use first and last spectrum lines as background level.
            # We should start and end the scan outside the trace anyway.
            bkg0 = self.scan_raw_data[0, :]
            bkg1 = self.scan_raw_data[-1, :]
            bkg = (bkg0 + bkg1) / 2
            # Tile background vector to a 2D matrix that can be subtracted from the data:
            bkg_m = np.tile(bkg, (self.scan_raw_data.shape[0], 1))
            proc_data = self.scan_raw_data - bkg_m
            self.logger.debug("Background image subtractged")
        else:
            proc_data = np.copy(self.scan_raw_data)
        # Normalization
        proc_data = proc_data / proc_data.max()
        self.logger.debug("Scan data normalized")
        # Thresholding
        thr = self.controller.analyse_params["threshold"]
        self.logger.debug("Threshold: {0}".format(thr))
        proc_thr = np.clip(proc_data - thr, 0, None)     # Thresholding
        self.logger.debug("Scan data thresholded to {0}".format(thr))
        # Filtering
        kernel = int(self.controller.analyse_params["median_kernel"])
        if kernel > 1:
            proc_thr = medfilt2d(proc_thr, kernel)
            self.logger.debug("Scan data median filtered with kernel size {0}".format(kernel))
        # self.logger.debug("proc_data {0}".format(proc_data))
        # self.logger.debug("proc_thr {0}".format(proc_thr))
        self.scan_proc_data = proc_thr

    def find_roi(self):
        """
        Find the ROI around the centroid of the processed scan image.
        :return:
        """
        self.logger.info("Running find_roi to center beam")
        I_tot = self.scan_proc_data.sum()
        xr = np.arange(self.scan_proc_data.shape[0])
        x_cent = (xr * self.scan_proc_data.sum(1)).sum() / I_tot
        yr = np.arange(self.scan_proc_data.shape[1])
        y_cent = (yr * self.scan_proc_data.sum(0)).sum() / I_tot
        self.logger.debug("Centroid position: {0:.1f} x {1:.1f}".format(x_cent, y_cent))
        xw = np.floor(np.minimum(x_cent - xr[0], xr[-1] - x_cent)).astype(np.int)
        yw = np.floor(np.minimum(y_cent - yr[0], yr[-1] - y_cent)).astype(np.int)
        x0 = np.floor(x_cent - xw).astype(np.int)
        x1 = np.floor(x_cent + xw).astype(np.int)
        y0 = np.floor(y_cent - yw).astype(np.int)
        y1 = np.floor(y_cent + yw).astype(np.int)
        self.scan_roi_data = self.scan_proc_data[x0:x1, y0:y1]
        self.controller.scan_raw_data = self.scan_raw_data
        self.controller.scan_proc_data = self.scan_proc_data
        self.controller.scan_roi_data = self.scan_roi_data

    def convert_data(self, use_roi=False):
        """
        Create the needed variables for the algorithm.
        :return:
        """
        self.logger.info("Converting scan data for quad scan algorithm")

    def fit_scan(self):
        self.logger.info("Fit quad scan")
        method = self.controller.analyse_params["method"]
        iterations = self.controller.analyse_params["iterations"]

    def retrieve_data(self, result):
        """
        Retrieve data from quad analysis analysis run. Run as a callback from a deferred.

        :param result: deferred result
        :return:
        """
        self.logger.info("Retrieving data from fit")

        self.controller.analysis_result["eps"] = None
        self.controller.analysis_result["beta"] = None
        self.controller.analysis_result["alpha"] = None
        self.controller.analysis_result["gamma"] = None
        self.d.callback(self.controller.analysis_result)

    def analysis_error(self, err):
        self.logger.error("Error in QuadScan analysis: {0}".format(err))
        self.d.errback(err)


def test_image_pool(a, b):
    print("Yo! {0}, {1}".format(a, b))
    return a


def process_image_func(image, k_ind, image_ind, threshold, roi_cent, roi_dim, cal=1, kernel=3, bpp=16):
    print("Processing image {0}, {1} in pool".format(k_ind, image_ind))
    t0 = time.time()
    x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
    y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])

    # Extract ROI and convert to double:
    pic_roi = np.double(image[x[0]:x[1], y[0]:y[1]])

    # Normalize pic to 0-1 range, where 1 is saturation:
    n = 2 ** bpp
    # if image.dtype == np.int32:
    #     n = 2 ** 16
    # elif image.dtype == np.uint8:
    #     n = 2 ** 8
    # else:
    #     n = 1

    # Median filtering:
    pic_roi = medfilt2d(pic_roi / n, kernel)

    # Threshold image
    pic_roi[pic_roi < threshold] = 0.0

    line_x = pic_roi.sum(0)
    line_y = pic_roi.sum(1)
    q = line_x.sum()  # Total signal (charge) in the image

    enabled = False
    l_x_n = np.sum(line_x)
    l_y_n = np.sum(line_y)
    # Enable point only if there is data:
    if l_x_n > 0.0:
        enabled = True
    x_v = cal[0] * np.arange(line_x.shape[0])
    y_v = cal[1] * np.arange(line_y.shape[0])
    x_cent = np.sum(x_v * line_x) / l_x_n
    sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
    y_cent = np.sum(y_v * line_y) / l_y_n
    sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)

    # Store processed data
    result = dict()
    result["k_ind"] = k_ind
    result["image_ind"] = image_ind
    result["pic_roi"] = pic_roi
    result["line_x"] = line_x
    result["line_y"] = line_y
    result["x_cent"] = x_cent
    result["y_cent"] = y_cent
    result["sigma_x"] = sigma_x
    result["sigma_y"] = sigma_y
    result["q"] = q
    result["enabled"] = enabled
    # result = [k_ind, image_ind, pic_roi, line_x, line_y, x_cent, y_cent, sigma_x, sigma_y, q, enabled]

    return result


if __name__ == "__main__":
    root = logging.getLogger()
    while len(root.handlers):
        root.removeHandler(root.handlers[0])

    f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
    fh = logging.StreamHandler()
    fh.setFormatter(f)
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)

    controller = QuadScanController()

    sh = qs.StateDispatcher(controller)
    sh.start()

    loaddir = "d:/Documents/Data/quadscan_data/2018-04-16_13-38-53_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
    loaddir2 = "d:/Documents/Data/quadscan_data/2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
    loaddir3 = "C:/Users/filip/Documents/workspace/emittancescansinglequad/saved-images/2018-04-16_13-38-53_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
    controller.set_parameter("load", "path", str(loaddir3))
    sh.send_command("load")




