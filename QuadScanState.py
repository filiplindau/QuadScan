# -*- coding:utf-8 -*-
"""
Created on May 18, 2018

@author: Filip Lindau

All data is in the controller object. The state object only stores data needed to keep track
of state progress, such as waiting deferreds.
When a state transition is started, a new state object for that state is instantiated.
The state name to class table is stored in a dict.

"""

import threading
import time
import logging
import os
import tango as tango
import numpy as np
from twisted_cut import defer
import TangoTwisted
import QuadScanController
from TangoTwisted import TangoAttributeFactory, defer_later
import PIL

logger = logging.getLogger("QuadScanState")
logger.setLevel(logging.DEBUG)
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class StateDispatcher(object):
    def __init__(self, controller):
        self.controller = controller
        self.stop_flag = False
        self.statehandler_dict = dict()
        self.statehandler_dict[StateUnknown.name] = StateUnknown
        self.statehandler_dict[StateDatabase.name] = StateDatabase
        self.statehandler_dict[StateDeviceConnect.name] = StateDeviceConnect
        self.statehandler_dict[StateSetupAttributes.name] = StateSetupAttributes
        self.statehandler_dict[StateIdle.name] = StateIdle
        self.statehandler_dict[StateScan.name] = StateScan
        self.statehandler_dict[StateAnalysis.name] = StateAnalysis
        self.statehandler_dict[StateLoad.name] = StateLoad
        self.statehandler_dict[StateSave.name] = StateSave
        self.statehandler_dict[StateFault] = StateFault
        self.current_state = StateDatabase.name
        self._state_obj = None
        self._state_thread = None

        self.logger = logging.getLogger("QuadScanController.StateDispatcher")
        self.logger.setLevel(logging.DEBUG)

    def statehandler_dispatcher(self):
        self.logger.info("Entering state handler dispatcher")
        prev_state = self.get_state()
        while self.stop_flag is False:
            # Determine which state object to construct:
            try:
                state_name = self.get_state_name()
                self.logger.debug("New state: {0}".format(state_name.upper()))
                self._state_obj = self.statehandler_dict[state_name](self.controller)
            except KeyError:
                state_name = "unknown"
                self.statehandler_dict[StateUnknown.name]
            self.controller.set_state(state_name, "")
            # Do the state sequence: enter - run - exit
            self._state_obj.state_enter(prev_state)
            self._state_obj.run()       # <- this should be run in a loop in state object and
            # return when it's time to change state
            new_state = self._state_obj.state_exit()
            # Set new state:
            self.set_state(new_state)
            prev_state = state_name
        self._state_thread = None

    def get_state(self):
        return self._state_obj

    def get_state_name(self):
        return self.current_state

    def set_state(self, state_name):
        try:
            self.logger.info("Switching state: {0} --> {1}".format(self.current_state.upper(),
                                                                   state_name.upper()))
            self.current_state = state_name
        except AttributeError:
            self.logger.debug("New state unknown. Got {0}, setting to UNKNOWN".format(state_name))
            self.current_state = "unknown"

    def send_command(self, msg, *args):
        self.logger.info("Sending command {0} to state {1}".format(msg, self.current_state))
        self._state_obj.check_message(msg, *args)

    def stop(self):
        self.logger.info("Stop state handler thread")
        self._state_obj.stop_run()
        self.stop_flag = True

    def start(self):
        self.logger.info("Start state handler thread")
        if self._state_thread is not None:
            self.stop()
        self._state_thread = threading.Thread(target=self.statehandler_dispatcher)
        self._state_thread.start()


class State(object):
    name = ""

    def __init__(self, controller):
        self.controller = controller    # type: QuadScanController.QuadScanController
        self.logger = logging.getLogger("QuadScanController.{0}".format(self.name.upper()))
        # self.logger.name =
        self.logger.setLevel(logging.WARNING)
        self.deferred_list = list()
        self.next_state = None
        self.cond_obj = threading.Condition()
        self.running = False

    def state_enter(self, prev_state=None):
        self.logger.info("Entering state {0}".format(self.name.upper()))
        with self.cond_obj:
            self.running = True
            self.controller.set_progress(0)

    def state_exit(self):
        self.logger.info("Exiting state {0}".format(self.name.upper()))
        for d in self.deferred_list:
            try:
                d.cancel()
            except defer.CancelledError:
                pass
        return self.next_state

    def run(self):
        self.logger.info("Entering run, run condition {0}".format(self.running))
        with self.cond_obj:
            if self.running is True:
                self.cond_obj.wait()
        self.logger.debug("Exiting run")

    def check_requirements(self, result):
        """
        If next_state is None: stay on this state, else switch state
        :return:
        """
        self.next_state = None
        return result

    def check_message(self, msg, *args):
        """
        Check message with condition object released and take appropriate action.
        The condition object is released already in the send_message function.

        -- This could be a message queue if needed...

        :param msg:
        :return:
        """
        pass

    def state_error(self, err):
        err_str = "Error in state {1}: {0}".format(err, self.name.upper())
        self.logger.error(err_str)
        self.controller.set_status(err_str)

    def get_name(self):
        return self.name

    def get_state(self):
        return self.name

    def send_message(self, msg):
        self.logger.info("Message {0} received".format(msg))
        with self.cond_obj:
            self.cond_obj.notify_all()
            self.check_message(msg)

    def stop_run(self):
        self.logger.info("Notify condition to stop run")
        with self.cond_obj:
            self.running = False
            self.logger.debug("Run condition {0}".format(self.running))
            self.cond_obj.notify_all()


class StateDeviceConnect(State):
    """
    Connect to tango devices needed.
    The names of the devices are stored in the controller.device_names list.
    Devices are stored as TangoAttributeFactories in controller.device_factory_dict

    """
    name = "device_connect"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.controller.device_factory_dict = dict()
        self.deferred_list = list()

    def state_enter(self, prev_state):
        State.state_enter(self, prev_state)
        self.controller.set_status("Connecting to devices.")
        sect_name = self.controller.get_parameter("scan", "section_name")
        quad_name = self.controller.get_parameter("scan", "quad_name")
        screen_name = self.controller.get_parameter("scan", "screen_name")
        if quad_name is None:
            # If there is no current quad / screen selected, take the first one for the section.
            # If there are none in the section, exit
            try:
                quad_name = self.controller.get_parameter("scan", "section_quads")[sect_name][0]["name"]
            except (IndexError, KeyError):
                self.next_state = "idle"
                self.stop_run()
        if screen_name is None:
            # If there is no current quad / screen selected, take the first one for the section.
            # If there are none in the section, exit
            try:
                screen_name = self.controller.get_parameter("scan", "section_screens")[sect_name][0]["name"]
            except (IndexError, KeyError):
                self.next_state = "idle"
                self.stop_run()
        dl = list()
        # set_section will connect to devices if needed. The connections are done asynchronously
        d = self.controller.set_section(sect_name, quad_name, screen_name)
        dl.append(d)
        self.deferred_list = dl
        d.addCallbacks(self.check_requirements, self.state_error)

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "idle"
        self.stop_run()
        return "idle"

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()


class StateSetupAttributes(State):
    """
    Setup attributes in the tango devices. Parameters stored in controller.setup_params
    Each key in setup_params is an attribute with the value as the value

    Device name is the name of the key in the controller.device_name dict (e.g. "camera").

    First the camera is put in ON state to be able to set certain attributes.
    When it is detected that the camera is in ON state the callback setup_attr is run,
    sending check_attributes on the attributes in the setup_params dict.

    """
    name = "setup_attributes"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.deferred_list = list()
        self.logger.setLevel(logging.DEBUG)

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        self.controller.set_status("Setting up device parameters.")
        self.logger.debug("Stopping camera before setting attributes")
        d_list = list()
        d1 = self.controller.send_command("stop", "camera", None)
        d2 = self.controller.check_attribute("state", "camera", tango.DevState.ON, timeout=3.0, write=False)
        self.logger.debug("Type d2: {0}".format(type(d2)))
        d2.addErrback(self.state_error)
        d_list.append(d1)
        d_list.append(d2)

        d = defer.DeferredList(d_list)
        self.logger.debug("Type deferred list: {0}".format(type(d)))
        d.addCallbacks(self.setup_attr, self.state_error)
        self.deferred_list.append(d)

    def setup_attr(self, result):
        self.logger.info("Entering setup_attr")
        # Go through all the attributes in the setup_attr_params dict and add
        # do check_attribute with write to each.
        # The deferreds are collected in a list that is added to a DeferredList
        # When the DeferredList fires, the check_requirements method is called
        # as a callback.
        dl = list()
        for key in self.controller.setup_attr_params:
            attr_name = key
            attr_value = self.controller.setup_attr_params[key]
            dev_name = "camera"
            try:
                self.logger.debug("Setting attribute {0} on device {1} to {2}".format(attr_name.upper(),
                                                                                      dev_name.upper(),
                                                                                      attr_value))
            except AttributeError:
                self.logger.debug("Setting attribute according to: {0}".format(attr_name))
            if attr_value is not None:
                d = self.controller.check_attribute(attr_name, dev_name, attr_value, period=0.3, timeout=2.0,
                                                    write=True)
            else:
                d = self.controller.read_attribute(attr_name, dev_name)
            d.addCallbacks(self.attr_check_cb, self.attr_check_eb)
            dl.append(d)

        # Create DeferredList that will fire when all the attributes are done:
        def_list = defer.DeferredList(dl)
        self.deferred_list.append(def_list)
        def_list.addCallbacks(self.check_requirements, self.state_error)

    def check_requirements(self, result):
        self.logger.info("Check requirements")
        # self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "running"
        self.stop_run()
        return result

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()

    def attr_check_cb(self, result):
        self.logger.info("Check attribute result: {0}".format(result))
        self.controller.camera_result[result.name.lower()] = result
        return result

    def attr_check_eb(self, err):
        self.logger.error("Check attribute ERROR: {0}".format(err))
        return err


class StateIdle(State):
    """
    Wait for time for a new scan or a command. Parameters stored in controller.idle_params
    idle_params["scan_interval"]: time in seconds between scans
    """
    name = "idle"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.t0 = time.time()
        self.logger.setLevel(logging.WARNING)

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        # Start camera:
        self.controller.set_status("Idle")
        # Start looping calls for monitored attributes
        self.start_looping_calls()

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.controller.set_status("Idle")
        return True

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        if err.type == defer.CancelledError:
            self.logger.info("Cancelled error, ignore")
        else:
            if self.running is True:
                self.controller.set_status("Error: {0}".format(err))
                self.stop_looping_calls()
                # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
                self.next_state = "unknown"
                self.stop_run()

    def check_message(self, msg, *args):
        if msg == "load":
            self.logger.debug("Message load... set next state and stop.")
            self.controller.idle_params["paused"] = True
            try:
                d = self.deferred_list[0]   # type: defer.Deferred
                d.cancel()
            except IndexError:
                self.logger.debug("Empty deferred list when cancelling")
            self.next_state = "load"
            self.stop_run()
        elif msg == "save":
            self.logger.debug("Message save... set next state and stop.")
            self.controller.idle_params["paused"] = True
            try:
                d = self.deferred_list[0]   # type: defer.Deferred
                d.cancel()
            except IndexError:
                self.logger.debug("Empty deferred list when cancelling")
            self.next_state = "save"
            self.stop_run()
        elif msg == "stop":
            self.logger.debug("Message stop... set next state.")
            for d in self.deferred_list:
                d.cancel()
            self.stop_looping_calls()
        elif msg == "process_images":
            self.logger.debug("Message process_images")
            d = self.controller.process_all_images()
            d.addErrback(self.state_error)
        elif msg == "fit_data":
            self.logger.debug("Message fit_data")
            d = defer.maybeDeferred(self.controller.fit_quad_data)
            d.addErrback(self.state_error)
        elif msg == "set_section":
            self.logger.debug("Message set section")
            self.controller.set_parameter("scan", "section_name", args[0])
            self.controller.set_parameter("scan", "quad_name", args[1])
            self.controller.set_parameter("scan", "screen_name", args[2])
#            self.next_state("device_connect")
#            self.stop_run()
        else:
            self.logger.warning("Unknown command {0}".format(msg))

    def stop_looping_calls(self):
        for lc in self.controller.looping_calls:
            # Stop looping calls (ignore callback):
            try:
                lc.stop()
            except Exception as e:
                self.logger.error("Could not stop looping call: {0}".format(e))
        self.controller.looping_calls = list()

    def start_looping_calls(self):
        self.logger.info("Starting looping calls")
        self.stop_looping_calls()
        quad_devices = self.controller.get_parameter("scan", "quad_device_names")
        screen_devices = self.controller.get_parameter("scan", "screen_device_names")
        interval = 1.0 / self.controller.idle_params["reprate"]

        dev_name = quad_devices["crq"]
        attr_name = "mainfieldcomponent"
        self.logger.debug("Starting looping call for {0}".format(attr_name))
        lc = TangoTwisted.LoopingCall(self.controller.read_attribute, attr_name, dev_name, use_tango_name=True)
        self.controller.looping_calls.append(lc)
        d = lc.start(interval)
        d.addCallbacks(self.controller.update_attribute, self.state_error)
        lc.loop_deferred.addCallback(self.controller.update_attribute)
        lc.loop_deferred.addErrback(self.state_error)

        dev_name = screen_devices["view"]
        attr_name = "image"
        self.logger.debug("Starting looping call for {0}".format(attr_name))
        lc = TangoTwisted.LoopingCall(self.controller.read_attribute, attr_name, dev_name, use_tango_name=True)
        self.controller.looping_calls.append(lc)
        d = lc.start(interval)
        d.addCallbacks(self.controller.update_attribute, self.state_error)
        lc.loop_deferred.addCallback(self.controller.update_attribute)
        lc.loop_deferred.addErrback(self.state_error)

    def update_attribute(self, result):
        self.logger.info("Updating result")
        try:
            self.logger.debug("Result for {0}: {1}".format(result.name, result.value))
        except AttributeError:
            return
        with self.controller.state_lock:
            self.controller.attr_result[result.name.lower()] = result


class StateScan(State):
    """
    Start a quad scan scan using scan_params parameters dict stored in the controller.
    scan_params["start_pos"]: initial quad magnet setting
    scan_params["step_size"]: quad magnet step size
    scan_params["end_pos"]: quad magnet end position
    # scan_params["dev_name"]: device name that runs the scan
    scan_params["scan_attr"]: name of attribute to scan
    scan_params["average"]: number of averages in each position
    """
    name = "scan"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.logger.setLevel(logging.INFO)

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        scan_dev_name = "quad"
        meas_dev_name = "screen"
        scan_dev_name = self.controller.get_parameter("scan", "quad_device_names")["crq"]
        meas_dev_name = self.controller.get_parameter("scan", "screen_device_names")["view"]
        scan_attr_name = "mainfieldcomponent"
        meas_attr_name = "image"
        start_pos = self.controller.scan_params["start_pos"]
        end_pos = self.controller.scan_params["end_pos"]
        step_size = self.controller.scan_params["step_size"]
        self.logger.info("Starting scan of {0} on {1}".format(scan_attr_name, scan_dev_name))
        self.controller.set_status("Scanning from {0} to {1} with step size {2}".format(start_pos, end_pos, step_size))
        scan = QuadScanController.Scan(self.controller, scan_attr_name, scan_dev_name, start_pos, end_pos, step_size,
                                       meas_dev_name, meas_attr_name)
        d = scan.start_scan()
        d.addCallbacks(self.check_requirements, self.state_error)
        self.deferred_list.append(d)

    def check_requirements(self, result):
        self.logger.debug("Check requirements result: {0}".format(result))
        self.controller.scan_result["pos_data"] = result[0]
        self.controller.scan_result["scan_data"] = result[1]
        self.controller.scan_result["start_time"] = result[2]
        self.next_state = "analyse"
        self.stop_run()
        return "analyse"

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()

    def check_message(self, msg):
        if msg == "pause":
            self.logger.debug("Message pause... stop.")
            self.controller.idle_params["paused"] = True
            d = self.deferred_list[0]   # type: defer.Deferred
            d.cancel()
            self.next_state = "idle"
            self.stop_run()
        elif msg == "cancel":
            self.logger.debug("Message cancel... set next state and stop.")
            self.controller.idle_params["paused"] = True
            d = self.deferred_list[0]   # type: defer.Deferred
            d.cancel()
            self.next_state = "idle"
            self.stop_run()
        elif msg == "scan":
            self.logger.debug("Message resume... continue scan")
            d = self.deferred_list[0]   # type: defer.Deferred
            d.cancel()
        elif msg == "process_images":
            self.logger.debug("Message process_images")
            d = self.controller.process_all_images()
            d.addErrback(self.state_error)
        elif msg == "fit_data":
            self.logger.debug("Message fit_data")
            d = defer.maybeDeferred(self.controller.fit_quad_data)
            d.addErrback(self.state_error)
        else:
            self.logger.warning("Unknown command {0}".format(msg))


class StateAnalysis(State):
    """
    Start quad analysis of latest scan data. Parameters are stored in controller.analyse_params
    analyse_params["method"]: FROG method (SHG, TG, ...)
    analyse_params["size"]: size of FROG trace (128, 256, ...)
    analyse_params["iterations"]: number of iterations to calculate
    analyse_params["roi"]: region of interest of the
    analyse_params["threshold"]: threshold level for the normalized data
    analyse_params["background_subtract"]: do background subtraction using start pos spectrum
    """
    name = "analysis"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.quad_analysis = None

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        self.controller.set_status("Analysing scan")
        self.logger.debug("Starting quad scan analysis")
        self.quad_analysis = QuadScanController.QuadScanAnalyse(self.controller)
        d = self.quad_analysis.start_analysis()
        self.deferred_list.append(d)
        d.addCallbacks(self.check_requirements, self.state_error)

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "idle"
        self.stop_run()
        return "idle"

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()


class StateLoad(State):
    """
    Load saved quad scan data. Parameters used for loading are stored in controller.load_params
    The loaded data is stored in controller.scan_result
    load_params["path"]: Directory path for images
    load_params["type"]: Source of the data files. "auto", "python", or "matlab". "python" is the current scan program.
                         "matlab" is the old maxiv quadscan  program. "auto" tries to determine from the directory
                         file structure which it is.
    """
    name = "load"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.logger.setLevel(logging.DEBUG)
        self.image_file_list = list()
        self.image_file_iterator = None

        self.old_raw_data = None
        self.old_proc_data = None

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        self.controller.set_status("Load scan data")
        self.logger.debug("Starting scan load")
        self.load_data()

    def load_data(self):
        old_dir = os.getcwd()
        load_dir = self.controller.load_params["path"]
        self.logger.debug("Loading from {0}".format(load_dir))
        try:
            os.chdir(load_dir)
        except OSError as e:
            e = "Change dir failed: {0}".format(e)
            os.chdir(old_dir)
            self.state_error(e)

        # See if there is a file called daq_info.txt
        filename = "daq_info.txt"
        if os.path.isfile(filename) is False:
            e = "daq_info.txt not found in {0}".format(load_dir)
            os.chdir(old_dir)
            self.state_error(e)

        self.logger.info("Loading Jason format data")
        data_dict = dict()
        with open(filename, "r") as file:
            while True:
                line = file.readline()
                if line == "" or line[0:5] == "*****":
                    break
                try:
                    key, value = line.split(":")
                    data_dict[key.strip()] = value.strip()
                except ValueError:
                    pass
        self.controller.set_parameter("scan", "beam_energy", np.double(data_dict["beam_energy"]))
        self.controller.set_parameter("scan", "quad_name", data_dict["quad"])
        self.controller.set_parameter("scan", "quad_length", np.double(data_dict["quad_length"]))
        self.controller.set_parameter("scan", "quad_screen_distance", np.double(data_dict["quad_2_screen"]))
        self.controller.set_parameter("scan", "screen_name", data_dict["screen"])
        self.controller.set_parameter("scan", "num_k_values", int(data_dict["num_k_values"]))
        self.controller.set_parameter("scan", "num_shots", int(data_dict["num_shots"]))
        self.controller.set_parameter("scan", "k_min", np.double(data_dict["k_min"]))
        self.controller.set_parameter("scan", "k_max", np.double(data_dict["k_max"]))
        px = data_dict["pixel_dim"].split(" ")
        self.logger.debug("pixel_dim: {0}".format(px))
        self.controller.set_parameter("scan", "pixel_size", [np.double(px[0]), np.double(px[1])])
        rc = data_dict["roi_center"].split(" ")
        self.controller.set_parameter("scan", "roi_center", [np.double(rc[1]), np.double(rc[0])])
        rd = data_dict["roi_dim"].split(" ")
        self.controller.set_parameter("scan", "roi_dim", [np.double(rd[1]), np.double(rd[0])])

        file_list = os.listdir(".")
        for file_name in file_list:
            if file_name.endswith(".png"):
                self.image_file_list.append(file_name)
        self.logger.debug("Found {0} images in directory".format(len(self.image_file_list)))
        self.image_file_iterator = iter(self.image_file_list)
        # Save old data:
        self.old_raw_data = self.controller.get_result("scan", "raw_data")
        self.old_proc_data = self.controller.get_result("scan", "proc_data")
        # Init new data structure:
        self.controller.set_result("scan", "raw_data",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "proc_data",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "line_data_x",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "line_data_y",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "x_cent",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "y_cent",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "sigma_x",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "sigma_y",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "k_data",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "enabled_data",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.controller.set_result("scan", "charge_data",
                                   [list() for i in range(self.controller.get_parameter("scan", "num_k_values"))])
        self.load_next_image(None)

    def load_next_image(self, result):
        self.logger.debug("Result type: {0}".format(type(result)))
        if result is not None:
            self.deferred_list.pop(0)
            name = result.filename.split("_")
            try:
                k_num = np.maximum(0, int(name[0]) - 1).astype(np.int)
                image_num = np.maximum(0, int(name[1]) - 1).astype(np.int)
                k_value = np.double(name[2])
            except ValueError:
                self.logger.error("Image filename wrong format: {0}".format(name))
                return False
            except IndexError:
                self.logger.error("Image filename wrong format: {0}".format(name))
                return False
            image_total = self.controller.scan_params["num_shots"] * self.controller.scan_params["num_k_values"]
            p = (k_num * self.controller.scan_params["num_shots"] + image_num + 1.0) / image_total
            self.logger.debug("Loading image {0}_{1}".format(k_num, image_num))
            self.controller.set_status("Loading image {0}_{1}".format(k_num, image_num))
            self.controller.set_progress(p)
            pic = np.array(result)
            self.logger.debug("np pic size: {0}".format(pic.shape))

            self.controller.add_raw_image(k_num, k_value, pic)
            self.controller.process_image(k_num, image_num)
        try:
            filename = self.image_file_iterator.next()
            self.logger.debug("Loading file {0}".format(filename))
            d = TangoTwisted.defer_to_thread(PIL.Image.open, filename)
            d.addCallbacks(self.load_next_image, self.state_error)
            self.deferred_list = [d]
            return True
        except StopIteration:
            self.logger.debug("No more files, stopping image read.")
            self.controller.fit_quad_data()
            self.controller.processing_done_signal.emit()
            self.check_requirements(True)
            return False

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "idle"
        self.controller.set_progress(0)
        self.stop_run()
        return "idle"

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()

    def check_message(self, msg):
        if msg == "cancel":
            self.logger.debug("Message cancel... set next state and stop.")
            self.controller.idle_params["paused"] = True
            try:
                d = self.deferred_list[0]   # type: defer.Deferred
                d.cancel()
            except IndexError:
                self.logger.debug("Empty deferred list when cancelling")
            # Restore data:
            self.set_result("scan", "raw_data", self.old_raw_data)
            self.set_result("scan", "proc_data", self.old_proc_data)
            self.next_state = "idle"
            self.stop_run()
        elif msg == "process_images":
            self.logger.debug("Message process_images")
            d = self.controller.process_all_images()
            d.addErrback(self.state_error)
        else:
            self.logger.warning("Unknown command {0}".format(msg))


class StateSave(State):
    """
    Save quad scan data. Parameters used for saving are stored in controller.save_params
    save_params["path"]: Directory path for images
    """
    name = "save"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.quad_analysis = None

    def state_enter(self, prev_state=None):
        State.state_enter(self, prev_state)
        self.controller.set_status("Save scan data")
        self.logger.debug("Starting scan save")
        self.check_requirements(None)

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "idle"
        self.stop_run()
        return "idle"

    def state_error(self, err):
        self.logger.error("Error: {0}".format(err))
        self.controller.set_status("Error: {0}".format(err))
        # If the error was DB_DeviceNotDefined, go to UNKNOWN state and reconnect later
        self.next_state = "unknown"
        self.stop_run()

    def check_message(self, msg):
        if msg == "cancel":
            self.logger.debug("Message cancel... set next state and stop.")
            self.controller.idle_params["paused"] = True
            try:
                d = self.deferred_list[0]   # type: defer.Deferred
                d.cancel()
            except IndexError:
                self.logger.debug("Empty deferred list when cancelling")
            self.next_state = "idle"
            self.stop_run()


class StateFault(State):
    """
    Handle fault condition.
    """
    name = "fault"

    def __init__(self, controller):
        State.__init__(self, controller)


class StateUnknown(State):
    """
    Limbo state.
    Wait and try to connect to devices.
    """
    name = "unknown"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.deferred_list = list()
        self.start_time = None
        self.wait_time = 5.0

    def state_enter(self, prev_state):
        self.logger.info("Starting state {0}".format(self.name.upper()))
        # self.controller.set_status("Waiting {0} s before trying to reconnect".format(self.wait_time))
        self.controller.set_status("Not connected to devices")
        self.start_time = time.time()
        # df = defer_later(self.wait_time, self.check_requirements, [None])
        # self.deferred_list.append(df)
        # df.addCallback(test_cb)
        self.running = True

    def check_requirements(self, result):
        self.logger.info("Check requirements result {0} for state {1}".format(result, self.name.upper()))
        self.next_state = "database"
        self.stop_run()

    def check_message(self, msg):
        if msg == "load":
            self.logger.debug("Message load... set next state and stop.")
            self.controller.idle_params["paused"] = True
            for d in self.deferred_list:
                d.cancel()
            self.next_state = "load"
            self.stop_run()
        elif msg == "save":
            self.logger.debug("Message save... set next state and stop.")
            self.controller.idle_params["paused"] = True
            for d in self.deferred_list:
                d.cancel()
            self.next_state = "save"
            self.stop_run()
        elif msg == "connect":
            self.logger.debug("Message connect... set next state and stop.")
            for d in self.deferred_list:
                d.cancel()
            self.next_state = "connect"
            self.stop_run()
        elif msg == "process_images":
            self.logger.debug("Message process_images")
            d = self.controller.process_all_images()
            d.addErrback(self.state_error)
        elif msg == "fit_data":
            self.logger.debug("Message fit_data")
            d = defer.maybeDeferred(self.controller.fit_quad_data)
            d.addErrback(self.state_error)
        else:
            self.logger.warning("Unknown command {0}".format(msg))


class StateDatabase(State):
    """
    Query database for devices.

    """
    name = "database"

    def __init__(self, controller):
        State.__init__(self, controller)
        self.deferred_list = list()
        self.start_time = None
        self.wait_time = 5.0

    def state_enter(self, prev_state):
        self.logger.info("Starting state {0}".format(self.name.upper()))
        self.controller.set_status("Checking database for devices")
        d = TangoTwisted.defer_to_thread(self.controller.populate_matching_sections)
        self.start_time = time.time()
        self.deferred_list.append(d)
        d.addCallbacks(self.check_requirements, self.state_error)
        self.running = True

    def check_requirements(self, result):
        self.logger.info("Check requirements result {0} for state {1}".format(result, self.name.upper()))
        self.next_state = "device_connect"
        self.stop_run()

    def check_message(self, msg):
        if msg == "load":
            self.logger.debug("Message load... set next state and stop.")
            self.controller.idle_params["paused"] = True
            for d in self.deferred_list:
                d.cancel()
            self.next_state = "load"
            self.stop_run()
        elif msg == "process_images":
            self.logger.debug("Message process_images")
            d = self.controller.process_all_images()
            d.addErrback(self.state_error)
        elif msg == "fit_data":
            self.logger.debug("Message fit_data")
            d = defer.maybeDeferred(self.controller.fit_quad_data)
            d.addErrback(self.state_error)
        else:
            self.logger.warning("Unknown command {0}".format(msg))

    def state_error(self, err):
        State.state_error(self, err)
        self.next_state = "unknown"
        self.stop_run()

def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


if __name__ == "__main__":
    fc = QuadScanController.QuadScanController()

    sh = StateDispatcher(fc)
    sh.start()
