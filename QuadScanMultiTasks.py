"""
Created 2020-12-03

Tasks for multiquad scanning.

@author: Filip Lindau
"""

import threading
import multiprocessing
import uuid
import logging
import time
import ctypes
import inspect
import PIL
from PIL import Image
import cv2
import numpy as np
from numpngw import write_png
import os
from collections import namedtuple
import pprint
import traceback
from scipy.signal import medfilt2d
from collections import namedtuple, OrderedDict
import json

from tasks.GenericTasks import *
from QuadScanDataStructs import *
from QuadScanTasks import TangoReadAttributeTask, TangoMonitorAttributeTask, TangoWriteAttributeTask, LoadQuadImageTask
from multiquad_lu import MultiQuadLookup


try:
    import PyTango as pt
except ImportError:
    try:
        import tango as pt
    except ModuleNotFoundError:
        pass

logging.getLogger("PngImagePlugin").setLevel(logging.WARNING)

logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class TangoMultiQuadScanTask(Task):
    def __init__(self, scan_param, device_handler, section_devices, name=None, timeout=None, trigger_dict=dict(), callback_list=list(),
                 read_callback=None):
        # type: (ScanParamMulti) -> None
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)

        self.logger.setLevel(logging.INFO)
        self.scan_param = scan_param
        self.device_handler = device_handler            # Device management
        self.section_devices = section_devices          # Dict cointaining structure for each section.
                                                        # The structure has lists of magnet and screen devices.
        self.magnet_names = None                        # List of magnet names in the current section
        self.crq_devices = None                         # List of crq devices (directly controlling the current in
                                                        # the magnets) in the current section
        self.beamenergy = None                          # Electron energy in section. Read from the magnet devices
        self.camera_name = None                         # Name of the camera in current section
        self.camera_device = None                       # Camera device
        self.roi = None                                 # Camera roi as 4 element list
        self.px_cal = None                              # Pixel calibration
        self.scan_result = None
        self.last_step_result = None
        self.read_callback = read_callback
        self.logger.debug("{0}: Scan parameters: {1}".format(self, scan_param))
        self.logger.debug("{0}: read_callback {1}".format(self, read_callback))
        self.mq_lookup = None                           # Multiquad scan object, using lookup table to find quad values
        self.pathname = self.scan_param.base_path       # Save path for images and daq_info

        self.last_shot_time = None
        self.current_step = None
        self.image_list = None
        self.image_p_list = None

    def action(self):
        self.logger.info("{0} starting multiquad scan of {1} targetting {2} x {3}. ".format(self, self.scan_param.section,
                                                                                            self.scan_param.target_sigma_x,
                                                                                            self.scan_param.target_sigma_y))
        self.logger.info("{0} measuring {1}. ".format(self, self.scan_param.screen_name))

        #
        # Scan sequence:
        # Initialization
        # 1. Load section look-up table
        # 2. Prepare quad magnet and screen connections
        # 3. Lock in target beam size
        # Scan
        # 1. Calculate target a,b for next step
        # 2. Solve quad values for this setting using lookup table
        # 3. Set new quad values
        # 4. Measure new beam size
        # 5. Call callbacks from this step

        k_list = list()
        timestamp_list = list()

        self.last_shot_time = time.time()
        self.current_step = 0
        self.image_list = list()
        self.image_p_list = list()

        failed_steps = 0

        if self.mq_lookup is None:
            self.mq_lookup = MultiQuadLookup()

        self.mq_lookup.reset_data()

        section = self.scan_param.section
        target_sigma_x = self.scan_param.target_sigma_x
        target_sigma_y = self.scan_param.target_sigma_y
        target_charge = self.scan_param.charge_ratio
        n_steps = self.scan_param.n_steps
        guess_alpha = self.scan_param.guess_alpha
        guess_beta = self.scan_param.guess_beta
        guess_eps_n = self.scan_param.guess_eps_n

        self.set_section(section)

        if self.scan_param.save:
            self.generate_daq_info()

        self.mq_lookup.start_scan(target_sigma_x, target_sigma_y, None, section,
                                  n_steps, guess_alpha, guess_beta, guess_eps_n)

        k_next = self.get_quad_magnets()
        current_step = 0

        # Loop through the scan
        while self.get_done_event().is_set() is False:

            k_current, image, image_p, timestamp, k_next = self.do_step(k_next)

            self.image_list.append(image)
            self.image_p_list.append(image_p)
            a_list = self.mq_lookup.a_list
            b_list = self.mq_lookup.b_list
            beta_list = self.mq_lookup.beta_list
            eps_list = self.mq_lookup.eps_n_list
            # self.last_step_result = ScanMultiStepResult(k_current, image, image_p, timestamp,
            #                                             a_list, b_list, beta_list, eps_list)
            self.last_step_result = {"k_list": self.mq_lookup.k_list, "image": image, "image_p": image_p,
                                     "timestamps": timestamp_list, "multiquad": self.mq_lookup}
            self.result = self.last_step_result

            if self.scan_param.save:
                self.save_image(image, self.current_step, k_current)

            k_list.append(k_current)
            timestamp_list.append(timestamp)
            current_step += 1

            # Step done, notify callbacks:
            if self.is_done() is False:
                self.logger.debug("{0} Calling {1} callbacks".format(self, len(self.callback_list)))
                for callback in self.callback_list:
                    callback(self)

            if current_step > self.scan_param.n_steps:
                self.logger.info("Scan completed.")
                self.event_done.set()

        self.scan_result = {"k_list": self.mq_lookup.k_list, "images": self.image_list,
                            "timestamps": timestamp_list, "multiquad": self.mq_lookup}
        self.result = self.scan_result

    def get_last_step_result(self):
        self.logger.debug("Last step result")
        return self.last_step_result

    def set_section(self, section="MS1"):
        """

        :param section:
        :return:
        """
        sect_quad_list = self.section_devices.sect_quad_dict[self.scan_param.section]
        # self.logger.debug("Section {0} devices:\n{1}".format(self.scan_param.section, sect_quad_list))
        self.magnet_names = [sect_quad.name for sect_quad in sect_quad_list]
        self.crq_devices = [self.device_handler.get_device(sect_quad.crq) for sect_quad in sect_quad_list]
        self.beamenergy = self.crq_devices[0].energy

        sect_screen_list = self.section_devices.sect_screen_dict[self.scan_param.section]
        camera_name = None
        for sect_screen in sect_screen_list:
            if self.scan_param.screen_name.lower() in sect_screen.name.lower():
                camera_name = sect_screen.name
        if camera_name is None:
            raise(ValueError("Camera name {0} not found in section {1}, only {2}".format(self.scan_param.screen_name,
                                                                                         self.scan_param.section,
                                                                                         sect_screen_list)))
            return

        self.camera_name = camera_name
        self.logger.info("Found screen {0}".format(self.camera_name))

        self.camera_device = self.device_handler.get_device(sect_screen.liveviewer)
        beam_device = self.device_handler.get_device(sect_screen.beamviewer)
        self.logger.info("Screen ROI: {0}".format(beam_device.roi))
        self.logger.info("Param ROI center: {0}, ROI dim: {1}".format(self.scan_param.roi_center, self.scan_param.roi_dim))
        roi = beam_device.roi
        rc = self.scan_param.roi_center
        rd = self.scan_param.roi_dim
        roi = [rc[0] - rd[0] / 2, rc[1] - rd[1] / 2, rd[0], rd[1]]
        # self.roi = [int(roi[0]), int(roi[2]), int(roi[1] - roi[0]), int(roi[3] - roi[2])]
        self.roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
        size = json.loads(beam_device.measurementruler)["size"]
        dx = beam_device.measurementrulerwidth
        self.px_cal = dx * 1e-3 / size[0]
        self.mq_lookup.set_section(section, load_file=True)
        self.logger.info("Section {0}: \n"
                         "Electron energy = {1:.3f} MeV\n"
                         "Pixel resolution = {2:.3f} um\n"
                         "ROI = {3} (l, t, w, h)".format(section, self.beamenergy * 1e-6, self.px_cal * 1e6, self.roi))

    def set_quad_magnets(self, k_list):

        for ind, dev in enumerate(self.crq_devices):
            dev.mainfieldcomponent = k_list[ind]
        time.sleep(self.magnet_delay)

    def get_quad_magnets(self):
        mag_task_list = list()
        # Read all quads and wait for the result
        for ind, dev in enumerate(self.crq_devices):
            read_task = TangoReadAttributeTask("mainfieldcomponent", dev,
                                               self.device_handler, name="read_mag_{0}".format(ind),
                                               timeout=self.timeout)
            mag_task_list.append(read_task)
        mag_bag_task = BagOfTasksTask(mag_task_list, name="measure_bag", timeout=self.timeout)
        mag_bag_task.start()
        mag_result = mag_bag_task.get_result(wait=True, timeout=self.timeout)
        self.logger.debug("Mag result: {0}".format(mag_result))
        k_current = [res.value for res in mag_result]
        return k_current

    def generate_daq_info(self):
        """
        Generate daq_info.txt file and AcceleratorParameters.
        Init quad_scan_data_scan with these parameters.
        :return: True is success
        """
        self.logger.info("Generating daq_info")

        base_path = self.scan_param.base_path

        try:
            s = "Multiquad_{0}_{1}".format(time.strftime("%Y-%m-%d_%H-%M-%S"), self.scan_param.section)
            self.pathname = os.path.join(base_path, s)
            os.makedirs(self.pathname)
        except OSError as e:
            self.logger.exception("Error creating directory: {0}".format(e))
            return False

        self.logger.info("Save path: {0}".format(self.pathname))

        roi_size = (self.roi[2], self.roi[3])
        roi_pos = (self.roi[0], self.roi[1])
        roi_center = [roi_pos[0] + roi_size[0] / 2.0, roi_pos[1] + roi_size[1] / 2.0]
        roi_dim = [roi_size[0], roi_size[1]]

        # Save daq_info_multi.txt:

        save_dict = OrderedDict()
        try:
            save_dict["main_dir"] = base_path
            save_dict["daq_dir"] = self.pathname
            for ind, quad in enumerate(self.magnet_names):
                save_dict["quad_{0}".format(ind)] = quad
                save_dict["quad_{0}_length".format(ind)] = "{0:.3f}".format(self.mq_lookup.quad_list[ind].length)
                save_dict["quad_{0}_pos".format(ind)] = "{0:.3f}".format(self.mq_lookup.quad_list[ind].position)
            save_dict["screen"] = self.camera_name
            save_dict["screen_pos"] = "{0:.3f}".format(self.mq_lookup.screen.position)
            save_dict["pixel_dim"] = "{0} {1}".format(self.px_cal, self.px_cal)
            save_dict["num_k_values"] = "{0}".format(self.scan_param.n_steps)
            save_dict["num_shots"] = "1"
            save_dict["k_min"] = "{0}".format(-self.mq_lookup.max_k)
            save_dict["k_max"] = "{0}".format(self.mq_lookup.max_k)
            val = roi_center
            save_dict["roi_center"] = "{0} {1}".format(val[0], val[1])
            val = roi_dim
            save_dict["roi_dim"] = "{0} {1}".format(val[0], val[1])
            save_dict["beam_energy"] = "{0:.3f}".format(self.beamenergy * 1e-6)
            save_dict["camera_bpp"] = 16
        except KeyError as e:
            msg = "Could not generate daq_info: {0}".format(e)
            self.logger.exception(msg)
            return False
        except IndexError as e:
            msg = "Could not generate daq_info: {0}".format(e)
            self.logger.exception(msg)
            return False

        full_name = os.path.join(self.pathname, "daq_info_multi.txt")
        with open(full_name, "w+") as f:
            for key, value in save_dict.items():
                s = "{0} : {1}\n".format(key.ljust(14, " "), value)
                f.write(s)
            f.write("***** Starting loop over quadrupole k-values *****\n")
            f.write("+------+-------+----------+----------+----------------------+\n")
            f.write("|  k   |  shot |    set   |   read   |        saved         |\n")
            f.write("|  #   |   #   |  k-value |  k-value |     image file       |\n")
        return True

    def save_image(self, image, step, k_values):
        self.logger.info("Saving image {0}".format(step))
        s = "_".join(["{0:.3f}".format(k) for k in k_values])
        filename = "{0}_{1:02d}_{2}.png".format(self.scan_param.section, step, s)
        self.logger.info("Filename {0}, pathname {1}".format(filename, self.pathname))
        full_name = os.path.join(self.pathname, filename)
        with open(full_name, "wb") as fh:
            try:
                write_png(fh, image.astype(np.uint16), filter_type=1)
            except Exception as e:
                self.logger.error("Image error: {0}".format(e))

    def do_step(self, k_next, save=True):
        t0 = time.time()
        # Prepare set of tasks for this scan position.
        # - Write new pos
        # - Wait until arrived at new pos (monitor attribute)
        # - Wait until new shot
        # - Read a set of measurements
        write_task_list = list()
        for write_ind, write_dev in enumerate(self.crq_devices):
            write_pos_task = TangoWriteAttributeTask("mainfieldcomponent",
                                                     write_dev,
                                                     self.device_handler,
                                                     k_next[write_ind],
                                                     name="write_pos_{0}".format(write_ind),
                                                     timeout=self.timeout)
            monitor_pos_task = TangoMonitorAttributeTask("mainfieldcomponent",
                                                         write_dev,
                                                         self.device_handler,
                                                         k_next[write_ind],
                                                         tolerance=self.scan_param.scan_pos_tol,
                                                         interval=self.scan_param.scan_pos_check_interval,
                                                         name="monitor_pos",
                                                         timeout=self.timeout)
            step_sequence_task = SequenceTask([write_pos_task, monitor_pos_task], name="step_seq")
            write_task_list.append(step_sequence_task)
        write_bag_task = BagOfTasksTask(write_task_list, name="write_bag", timeout=self.timeout)

        # Should wait for a new image here instead of fixed delay
        delay_task = DelayTask(self.scan_param.measure_interval, name="image_delay")

        read_task = TangoReadAttributeTask("image", self.camera_device,
                                           self.device_handler, name="read_image", timeout=self.timeout)
        rep_task = RepeatTask(read_task, self.scan_param.measure_number, self.scan_param.measure_interval,
                              name="rep_image", timeout=self.timeout)
        step_sequence_task = SequenceTask([write_bag_task, delay_task, rep_task], name="step_seq")
        step_sequence_task.start()
        res = step_sequence_task.get_result(wait=True, timeout=self.timeout)

        k_current = [res.get_result(wait=False)[1].value for res in write_task_list]
        # How to deal with multiple images?
        image = res[2][0].value

        sigma_x, sigma_y, image_p = self.process_image(image, self.scan_param.charge_ratio)
        self.image_list.append(image)
        self.image_p_list.append(image_p)
        charge = image_p.sum()
        self.logger.info("Step {0}: sigma_x={1:.3f} mm, sigma_y={2:.3f} mm".format(self.current_step,
                                                                                   sigma_x*1e3, sigma_y*1e3))
        if save:
            self.save_image(image, self.current_step, k_current)

        timestamp = res[2][0].time

        k_next = self.mq_lookup.scan_step(sigma_x, sigma_y, charge, k_current[0:4])
        if k_next is not None:
            if len(k_current) > 4:
                k_set = k_current
                k_set[0:4] = k_next
            else:
                k_set = k_next
        # time.sleep(self.shot_delay)
        self.last_shot_time = time.time()
        self.current_step += 1
        self.logger.debug("Step time: {0:.3f} s".format(time.time() - t0))
        return k_current, image, image_p, timestamp, k_set

    def process_image(self, image, keep_charge_ratio=0.95):
        t0 = time.time()
        self.logger.info("Process image roi: {0}\n image size: {1}x{2}".format(self.roi, image.shape[0], image.shape[1]))
        image_roi = np.double(image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]])

        # Median filter and background subtraction:
        # Improve background subtraction!
        image_p = medfilt2d(image_roi, 5)
        bkg = image_p[0:50, 0:50].max() * 2
        image_p[image_p < bkg] = 0

        # Find charge to keep:
        n_bins = np.unique(image_p.flatten()).shape[0]
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(image_p, n_bins)
        # Cumulative charge as function of pixel value:
        hq = (h[0]*h[1][:-1]).cumsum()
        # Normalize:
        hq = hq / np.float(hq.max())
        # Find index above which the desired charge ratio is:
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        # Background level:
        d = (h[1][1] - h[1][0])/2.0
        th_q = h[1][th_ind] - d
        image_p[image_p < th_q] = 0.0

        x_v = np.arange(image_p.shape[1])
        y_v = np.arange(image_p.shape[0])
        w0 = image_p.sum()
        x0 = (image_p.sum(0) * x_v).sum() / w0
        y0 = (image_p.sum(1) * y_v).sum() / w0
        sigma_x = self.px_cal * np.sqrt((image_p.sum(0) * (x_v - x0) ** 2).sum() / w0)
        sigma_y = self.px_cal * np.sqrt((image_p.sum(1) * (y_v - y0) ** 2).sum() / w0)
        self.logger.debug("Process image \n\n"
                          "Roi size {0}x{1}\n"
                          "sigma {3:.3f} x {4:.3f} mm\n\n"
                          "Time: {2:.3f} ms".format(image_roi.shape[1], image_roi.shape[0],
                                                    1e3 * (time.time() - t0), sigma_x * 1e3, sigma_y * 1e3))
        return sigma_x, sigma_y, image_p

    def do_scan(self, section="MS1", n_steps=16, save=True, alpha0=0, beta0=10, eps_n_0=2e-6):
        self.n_steps = n_steps
        self.set_section(section)
        # self.set_quad_magnets([4.9, -4.3, 1.4, 0.5])

        if save:
            self.generate_daq_info()

        self.mq_lookup.start_scan(self.sigma_target_x, self.sigma_target_y, self.charge, section, self.n_steps,
                           alpha0, beta0, eps_n_0)
        self.current_step = 0

        for step in range(self.n_steps):
            self.do_step(save)


class LoadMultiQuadScanDirTask(Task):
    def __init__(self, quadscandir, process_now=True, threshold=None, kernel_size=3, process_exec_type="thread",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.INFO)

        self.pathname = quadscandir
        self.process_now = process_now
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.processed_image_list = list()
        self.acc_params = None
        self.task_seq = None
        self.update_image_flag = False          # Set this if read images should be sent to callbacks

    def action(self):
        load_dir = self.pathname
        self.logger.debug("{1}: Loading from {0}".format(load_dir, self))
        try:
            os.listdir(load_dir)
        except OSError as e:
            e = "List dir failed: {0}".format(e)
            self.result = e
            self.logger.error(e)
            self.cancel()
            return

        # See if there is a file called daq_info.txt
        filename = "daq_info_multi.txt"
        if os.path.isfile(os.path.join(load_dir, filename)) is False:
            e = "daq_info_multi.txt not found in {0}".format(load_dir)
            self.result = e
            self.logger.error(e)
            self.cancel()
            return

        logger.info("{0}: Loading Jason format data".format(self))
        data_dict = dict()
        quad_list = list()
        with open(os.path.join(load_dir, filename), "r") as daq_file:
            while True:
                line = daq_file.readline()
                if line == "" or line[0:5] == "*****":
                    break
                try:
                    key, value = line.split(":")
                    data_dict[key.strip()] = value.strip()
                except ValueError:
                    pass
        px = data_dict["pixel_dim"].split(" ")

        data_dict["pixel_size"] = [np.double(px[0]), np.double(px[1])]
        rc = data_dict["roi_center"].split(" ")
        data_dict["roi_center"] = [np.double(rc[1]), np.double(rc[0])]
        rd = data_dict["roi_dim"].split(" ")
        data_dict["roi_dim"] = [np.double(rd[1]), np.double(rd[0])]
        try:
            data_dict["bpp"] = np.int(data_dict["bpp"])
        except KeyError:
            data_dict["bpp"] = 16
            self.logger.debug("{1} Loaded data_dict: \n{0}".format(pprint.pformat(data_dict), self))

        self.acc_params = AcceleratorParameters(electron_energy=float(data_dict["beam_energy"]),
                                                quad_length=float(data_dict["quad_0_length"]),
                                                quad_screen_dist=float(data_dict["screen_pos"]) -
                                                                 float(data_dict["quad_0_pos"]),
                                                k_max=float(data_dict["k_max"]),
                                                k_min=float(data_dict["k_min"]),
                                                num_k=int(data_dict["num_k_values"]),
                                                num_images=int(data_dict["num_shots"]),
                                                cal=data_dict["pixel_size"],
                                                quad_name=data_dict["quad"],
                                                screen_name=data_dict["screen"],
                                                roi_center=data_dict["roi_center"],
                                                roi_dim=data_dict["roi_dim"])

        n_images = self.acc_params.num_k * self.acc_params.num_images

        file_list = os.listdir(load_dir)
        image_file_list = list()
        load_task_list = list()         # List of tasks, each loading an image. Loading should be done in sequence
                                        # as this in not sped up by paralellization
        for file_name in file_list:
            if file_name.endswith(".png"):
                image_file_list.append(file_name)
                # t = LoadQuadImageTask(file_name, load_dir, name=file_name,
                #                       callback_list=[self.image_processor.process_image])
                t = LoadQuadImageTask(file_name, load_dir, name=file_name,
                                      callback_list=[self.processed_image_done])
                t.logger.setLevel(logging.WARNING)
                load_task_list.append(t)

        self.logger.debug("{1} Found {0} images in directory".format(len(image_file_list), self))
        with self.lock:
            self.update_image_flag = True
        self.task_seq = SequenceTask(load_task_list, name="load_seq")
        self.task_seq.start()
        # Wait for image sequence to be done reading:
        image_list = self.task_seq.get_result(wait=True)
        if self.task_seq.is_cancelled():
            self.logger.error("Load image error: {0}".format(image_list))

        with self.lock:
            self.update_image_flag = False

        self.result = QuadScanData(self.acc_params, image_list, self.processed_image_list)

    def processed_image_done(self, load_quadimage_task):
        with self.lock:
            go = self.update_image_flag

            if go:
                quad_image = load_quadimage_task.get_result(wait=False)    # type: QuadImage

                pic_roi = np.zeros((int(self.acc_params.roi_dim[0]), int(self.acc_params.roi_dim[1])), dtype=np.float32)
                line_x = pic_roi.sum(0)
                line_y = pic_roi.sum(1)
                proc_image = ProcessedImage(k_ind=quad_image.k_ind, k_value=quad_image.k_value, image_ind=quad_image.image_ind,
                                            pic_roi=pic_roi, line_x=line_x, line_y=line_y, x_cent=pic_roi.shape[0]/2,
                                            y_cent=pic_roi.shape[1]/2, sigma_x=0.0, sigma_y=0.0,
                                            q=0, threshold=self.threshold, enabled=True)
                self.processed_image_list.append(proc_image)

                if isinstance(quad_image, Exception):
                    self.logger.error("{0}: Found error in processed image: {1}".format(self, quad_image))
                    return

                self.result = quad_image
                for callback in self.callback_list:
                    callback(self)

    def cancel(self):
        if self.task_seq is not None:
            self.task_seq.cancel()
        Task.cancel(self)