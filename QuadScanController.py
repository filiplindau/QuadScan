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
import QuadScanState as qs
import numpy as np
from scipy.signal import medfilt2d

from QuadScanEmittanceCalculation import QuadScanEmittanceCalculation

logger = logging.getLogger("QuadScanController")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class QuadScanController(object):
    def __init__(self, quad_name=None, screen_name=None, start=False):
        """
        Controller for running a scanning Frog device. Communicates with a spectrometer and a motor.


        :param quad_name: Tango device name of the quad used for scanning
        :param screen_name: Tango device name of the camera used for capturing images of the screen
        """
        self.device_names = dict()
        if quad_name is not None:
            self.device_names["quad"] = quad_name
        if screen_name is not None:
            self.device_names["camera"] = screen_name

        self.device_factory_dict = dict()

        self.setup_attr_params = dict()
        # self.setup_attr_params["speed"] = ("motor", "speed", 50.0)

        self.looping_calls = list()

        self.attr_result = dict()
        self.attr_result["image"] = None
        self.attr_result["magnet"] = None

        self.idle_params = dict()
        self.idle_params["reprate"] = 2.0
        self.idle_params["camera_attr"] = "image"
        self.idle_params["paused"] = False

        self.scan_params = dict()
        self.scan_params["k_min"] = 8.6
        self.scan_params["k_max"] = 8.75
        self.scan_params["num_k_values"] = 1
        self.scan_params["num_shots"] = 1
        self.scan_params["quad_name"] = ""
        self.scan_params["quad_length"] = 1.0
        self.scan_params["quad_screen_dist"] = 1.0
        self.scan_params["screen_name"] = ""
        self.scan_params["pixel_size"] = 1.0
        self.scan_params["beam_energy"] = 1.0
        self.scan_params["roi_center"] = [1.0, 1.0]
        self.scan_params["roi_dim"] = [1.0, 1.0]
        # self.scan_params["dev_name"] = "motor"

        self.scan_result = dict()
        self.scan_result["k_data"] = None
        self.scan_result["raw_data"] = None
        self.scan_result["proc_data"] = None
        self.scan_result["scan_data"] = None
        self.scan_result["start_time"] = None
        self.scan_raw_data = None
        self.scan_proc_data = None
        self.scan_roi_data = None

        self.analysis_result = dict()
        self.analysis_result["eps"] = None
        self.analysis_result["beta"] = None
        self.analysis_result["gamma"] = None
        self.analysis_result["alpha"] = None

        self.analyse_params = dict()
        self.analyse_params["method"] = "GP"
        self.analyse_params["iterations"] = 70
        self.analyse_params["roi"] = "full"
        self.analyse_params["threshold"] = 0.02
        self.analyse_params["median_kernel"] = 1
        self.analyse_params["background_subtract"] = True

        self.load_params = dict()
        self.load_params["path"] = "."

        self.save_params = dict()

        self.logger = logging.getLogger("QuadScanController.Controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("QuadScanController.__init__")

        self.state_lock = threading.Lock()
        self.status = ""
        self.state = "unknown"
        self.state_notifier_list = list()       # Methods in this list will be called when the state
        # or status message is changed
        self.progress = 0
        self.progress_notifier_list = list()

        if start is True:
            for key in self.device_names:
                self.device_factory_dict[key] = TangoAttributeFactory(self.device_names[key])
                self.device_factory_dict[key].startFactory()

    def read_attribute(self, name, device_name):
        self.logger.info("Read attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[device_name]]
            d = factory.buildProtocol("read", name)
        else:
            er = ValueError("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def write_attribute(self, name, device_name, data):
        self.logger.info("Write attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[device_name]]
            d = factory.buildProtocol("write", name, data)
        else:
            er = ValueError("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
            self.logger.error(er)
            d = defer.Deferred()
            d.errback(er)
        return d

    def send_command(self, name, device_name, data):
        self.logger.info("Send command \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[device_name]]
            d = factory.buildProtocol("command", name, data)
        else:
            self.logger.error("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
            err = AttributeError("Device {0} not used. "
                                 "The device is not in the list of devices used by this controller".format(device_name))
            d = defer.Deferred()
            d.errback(err)

        return d

    def defer_later(self, delay, delayed_callable, *a, **kw):
        self.logger.info("Calling {0} in {1} seconds".format(delayed_callable, delay))

        def defer_later_cancel(deferred):
            delayed_call.cancel()

        d = defer.Deferred(defer_later_cancel)
        d.addCallback(lambda ignored: delayed_callable(*a, **kw))
        delayed_call = threading.Timer(delay, d.callback, [None])
        delayed_call.start()
        return d

    def check_attribute(self, attr_name, dev_name, target_value, period=0.3, timeout=1.0, tolerance=None, write=True):
        """
        Check an attribute to see if it reaches a target value. Returns a deferred for the result of the
        check.
        Upon calling the function the target is written to the attribute if the "write" parameter is True.
        Then reading the attribute is polled with the period "period" for a maximum number of retries.
        If the read value is within tolerance, the callback deferred is fired.
        If the read value is outside tolerance after retires attempts, the errback is fired.
        The maximum time to check is then period x retries

        :param attr_name: Tango name of the attribute to check, e.g. "position"
        :param dev_name: Tango device name to use, e.g. "gunlaser/motors/zaber01"
        :param target_value: Attribute value to wait for
        :param period: Polling period when checking the value
        :param timeout: Time to wait for the attribute to reach target value
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :return: Deferred that will fire depending on the result of the check
        """
        self.logger.info("Check attribute \"{0}\" on \"{1}\"".format(attr_name, dev_name))
        if dev_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[dev_name]]
            d = factory.buildProtocol("check", attr_name, None, write=write, target_value=target_value,
                                      tolerance=tolerance, period=period, timeout=timeout)
        else:
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

    def set_progress(self, progress):
        self.logger.debug("Setting progress to {0}".format(progress))
        with self.state_lock:
            self.progress = progress
        self.logger.debug("Notifying progress listeners")
        for m in self.progress_notifier_list:
            m(progress)

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
            elif state_name == "analyse":
                self.analyse_params[param_name] = value
            elif state_name == "scan":
                self.scan_params[param_name] = value
            elif state_name == "idle":
                self.idle_params[param_name] = value

    def get_parameter(self, state_name, param_name):
        self.logger.debug("Getting parameter {0}.{1}:".format(state_name, param_name))
        with self.state_lock:
            try:
                if state_name == "load":
                    value = self.load_params[param_name]
                elif state_name == "save":
                    value = self.save_params[param_name]
                elif state_name == "analyse":
                    value = self.analyse_params[param_name]
                elif state_name == "scan":
                    value = self.scan_params[param_name]
                elif state_name == "idle":
                    value = self.idle_params[param_name]
                else:
                    value = None
            except KeyError:
                value = None
        self.logger.debug("Value {0}".format(value))
        return value

    def set_result(self, state_name, result_name, value):
        self.logger.debug("Setting result {0}.{1}:".format(state_name, result_name))
        with self.state_lock:
            if state_name == "analyse":
                self.analyse_result[result_name] = value
            elif state_name == "scan":
                self.scan_result[result_name] = value

    def get_result(self, state_name, result_name):
        self.logger.debug("Getting result {0}.{1}:".format(state_name, result_name))
        with self.state_lock:
            try:
                if state_name == "analyse":
                    value = self.analyse_result[result_name]
                elif state_name == "scan":
                    value = self.scan_result[result_name]
                else:
                    value = None
            except KeyError:
                value = None
        return value

    def process_image(self, image):
        self.logger.info("Processing image")
        th = self.get_parameter("analyse", "threshold")
        roi_cent = self.get_parameter("scan", "roi_center")
        roi_dim = self.get_parameter("scan", "roi_dim")
        x = np.array([int(roi_cent[0] - roi_dim[0]), int(roi_cent[0] + roi_dim[1])])
        y = np.array([int(roi_cent[1] - roi_dim[0]), int(roi_cent[1] + roi_dim[1])])
        self.logger.debug("Threshold: {0}\nROI: {1}-{2}, {3}-{4}".format(th, x[0], x[1], y[0], y[1]))
        pic_roi = image[x[0]:x[1], y[0]:y[1]]
        # pic_roi[pic_roi < th] = 0
        return pic_roi


class Scan(object):
    """
    Run a scan of one attribute while measuring another
    """
    def __init__(self, controller, scan_attr_name, scan_dev_name, start_pos, stop_pos, step,
                 meas_attr_name, meas_dev_name):
        self.controller = controller    # type: QuadScanController.QuadScanController
        self.scan_attr = scan_attr_name
        self.scan_dev = scan_dev_name
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.step = step
        self.meas_attr = meas_attr_name
        self.meas_dev = meas_dev_name

        self.current_pos = None
        self.pos_data = []
        self.meas_data = []
        self.scan_start_time = None
        self.scan_arrive_time = time.time()
        self.data_time = time.time()
        self.status_update_time = time.time()
        self.status_update_interval = 1.0

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
                                             0.1, 10.0, tolerance=tol, write=True)
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
        self.logger.info("Scan step")
        tol = self.step * 0.1
        scan_pos = self.current_pos + self.step
        if scan_pos > self.stop_pos or self.cancel_flag is True:
            self.scan_done()
            return self.d

        d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos,
                                             0.1, 3.0, tolerance=tol, write=True)
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
        try:
            self.current_pos = result.value
        except AttributeError:
            self.logger.error("Scan arrive not returning attribute: {0}".format(result))
            return False

        self.logger.info("Scan arrive at pos {0}".format(self.current_pos))
        t = time.time()
        if t - self.status_update_time > self.status_update_interval:
            status = "Scanning from {0} to {1} with step size {2}\n" \
                     "Position: {3}".format(self.start_pos, self.stop_pos, self.step, self.current_pos)
            self.controller.set_status(status)
            self.status_update_time = t

        # After arriving, check the time against the last read data. Wait for a time so that the next
        # data frame should be there.
        self.scan_arrive_time = t
        try:
            wait_time = (self.scan_arrive_time - self.data_time) % self.controller.idle_params["reprate"]
        except KeyError:
            wait_time = 0.1
        d0 = defer_later(wait_time, self.meas_read_issue)
        d0.addErrback(self.scan_error_cb)
        self.logger.debug("Scheduling read in {0} s".format(wait_time))

        return True

    def meas_read_issue(self):
        """
        Called after scan_arrive to read a new data point.
        :return:
        """
        self.logger.info("Reading measurement")
        d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
        d0.addCallbacks(self.meas_scan_store, self.scan_error_cb)
        return True

    def meas_scan_store(self, result):
        self.logger.debug("Meas scan result: {0}".format(result.value))
        if result.time.totime() <= self.scan_arrive_time:
            self.logger.debug("Old data. Wait for new.")
            t = time.time() - result.time.totime()
            if t > 2.0:
                self.logger.error("Timeout waiting for new data. {0} s elapsed".format(t))
                self.scan_error_cb(RuntimeError("Timeout waiting for new data"))
                return False
            d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
            d0.addCallbacks(self.meas_scan_store, self.scan_error_cb)
            return False
        self.data_time = result.time.totime()
        measure_value = result.value

        self.logger.debug("Measure at scan pos {0} result: {1}".format(self.current_pos, measure_value))
        self.meas_data.append(measure_value)
        self.pos_data.append(self.current_pos)
        self.scan_step()
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
        Retrieve data from frog analysis run. Run as a callback from a deferred.
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


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


def test_timeout(result):
    logger.warning("TIMEOUT returned {0}".format(result))


if __name__ == "__main__":
    # fc = QuadScanController("sys/tg_test/1", "gunlaser/motors/zaber01")
    fc = QuadScanController("gunlaser/devices/spectrometer_frog", "gunlaser/motors/zaber01")
    # time.sleep(0)
    # dc = fc.check_attribute("position", "motor", 7.17, 0.1, 0.5, 0.001, True)
    # dc.addCallbacks(test_cb, test_err)
    # time.sleep(1.0)
    # dc.addCallback(lambda _: TangoTwisted.DelayedCallReactorless(2.0 + time.time(),
    #                                                              fc.start_scan, ["position", 5, 10, 0.5,
    #                                                                              "double_scalar"]))
    # scan = TangoTwisted.Scan(fc, "position", "motor", 5, 10, 0.5, "double_scalar", "spectrometer")
    # ds = scan.start_scan()
    # ds.addCallback(test_cb)

    sh = qs.StateDispatcher(fc)
    sh.start()

    # da = fc.read_attribute("double_scalar", "motor")
    # da.addCallbacks(test_cb, test_err)
    # da = fc.write_attribute("double_scalar_w", "motor", 10)
    # da.addCallbacks(test_cb, test_err)

    # da = fc.defer_later(3.0, fc.read_attribute, "short_scalar", "motor")
    # da.addCallback(test_cb, test_err)

    # lc = LoopingCall(fc.read_attribute, "double_scalar_w", "motor")
    # dlc = lc.start(1)
    # dlc.addCallbacks(test_cb, test_err)
    # lc.loop_deferred.addCallback(test_cb)

    # clock = ClockReactorless()
    #
    # defcond = DeferredCondition("result.value>15", fc.read_attribute, "double_scalar_w", "motor")
    # dcd = defcond.start(1.0, timeout=3.0)
    # dcd.addCallbacks(test_cb, test_err)

