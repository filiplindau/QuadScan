"""
Created 2018-10-24

Tasks for async sequencing.

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

from tasks.GenericTasks import *
from QuadScanDataStructs import *
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
logger.setLevel(logging.INFO)


class TangoDeviceConnectTask(Task):
    def __init__(self, device_name, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
        # Exceptions are caught in the parent run thread.
        self.logger.debug("Connecting to {0}".format(self.device_name))
        try:
            dev = pt.DeviceProxy(self.device_name)
        except pt.DevFailed as e:
            self.result = e
            self.cancel()
            return
        self.result = dev


class TangoReadAttributeTask(Task):
    def __init__(self, attribute_name, device_name, device_handler, name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list(), ignore_tango_error=True):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler
        self.ignore_tango_error = ignore_tango_error

        self.logger.setLevel(logging.WARNING)

    def action(self):
        self.logger.info("{0} reading {1} on {2}. ".format(self, self.attribute_name, self.device_name))
        try:
            dev = self.device_handler.get_device(self.device_name)
        except pt.DevFailed as e:
            self.logger.error("{0}: Could not connect. {1}".format(self, e))
            self.result = e
            self.cancel()
            return
        retries = 0
        while retries < 3:
            try:
                attr = dev.read_attribute(self.attribute_name)
                break
            except AttributeError as e:
                self.logger.exception("{0}: Attribute error reading {1} on {2}: ".format(self,
                                                                                         self.attribute_name,
                                                                                         self.device_name))
                attr = None
                self.result = e
                self.cancel()
                return
            except pt.DevFailed as e:
                self.logger.exception("{0}: Tango error reading {1} on {2}: ".format(self,
                                                                                     self.attribute_name,
                                                                                     self.device_name))
                attr = None
                self.result = e
                if not self.ignore_tango_error:
                    self.cancel()
                    return

            retries += 1
        self.result = attr


class TangoWriteAttributeTask(Task):
    def __init__(self, attribute_name, device_name, device_handler, value, name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler
        self.value = value
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} writing {1} to {2} on {3}. ".format(self,
                                                                  self.value,
                                                                  self.attribute_name,
                                                                  self.device_name))
        try:
            dev = self.device_handler.get_device(self.device_name)
        except pt.DevFailed as e:
            self.logger.error("{0}: Could not connect. {1}".format(self, e))
            self.result = e
            self.cancel()
            return
        try:
            res = dev.write_attribute(self.attribute_name, self.value)
        except pt.DevFailed as e:
            self.logger.error("{0}: Could not write attribute {1} with {2}. {3}".format(self, self.attribute_name,
                                                                                        self.value, e))
            self.result = e
            self.cancel()
            return
        self.result = res


class TangoCommandTask(Task):
    def __init__(self, command_name, device_name, device_handler, value=None, name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.command_name = command_name
        self.device_handler = device_handler
        self.value = value
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} sending command {1} with {2} on {3}. ".format(self,
                                                                            self.command_name,
                                                                            self.value,
                                                                            self.device_name))
        try:
            dev = self.device_handler.get_device(self.device_name)
        except pt.DevFailed as e:
            self.logger.error("{0}: Could not connect. {1}".format(self, e))
            self.result = e
            self.cancel()
            return
        try:
            res = dev.command_inout(self.command_name, self.value)
        except pt.DevFailed as e:
            self.logger.error("{0}: Could not write command {1} with {2}. {3}".format(self, self.command_name,
                                                                                      self.value, e))
            self.result = e
            self.cancel()
            return
        self.result = res


class TangoMonitorAttributeTask(Task):
    def __init__(self, attribute_name, device_name, device_handler, target_value, interval=0.5, tolerance=0.01,
                 tolerance_type="abs", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler
        self.target_value = target_value
        self.interval = interval
        self.tolerance = tolerance
        if tolerance_type == "rel":
            self.tol_div = self.target_value
            if self.tol_div == 0.0:
                raise AttributeError("Target value = 0 with relative tolerance type not possible.")
        else:
            self.tol_div = 1.0
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} monitor reading {1} from {2}. ".format(self, self.attribute_name, self.device_name))
        current_value = float("inf")
        read_task = TangoReadAttributeTask(self.attribute_name, self.device_name,
                                           self.device_handler, timeout=self.timeout,
                                           name="read_monitor_{0}".format(self.attribute_name))
        wait_time = -1
        while abs((current_value - self.target_value) / self.tol_div) > self.tolerance:
            if wait_time > 0:
                time.sleep(wait_time)
            t0 = time.time()
            read_task.start()
            current_value = read_task.get_result(wait=True, timeout=self.timeout).value
            if read_task.is_cancelled() is True:
                self.result = current_value
                self.cancel()
                return
            t1 = time.time()
            wait_time = self.interval - (t1 - t0)
        self.logger.info("{0} monitor completed: Target {1:.3f}, Measured {2:.3f}.".format(self, self.target_value, current_value))
        self.result = read_task.get_result(wait=False)


class LoadQuadImageTask(Task):
    def __init__(self, image_name, path=".", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.image_name = image_name
        self.path = path
        self.logger.setLevel(logging.WARNING)

    def action(self):
        self.logger.info("{0} entering action. Loading file {1}".format(self, self.image_name))
        name = self.image_name.split("_")
        try:
            k_ind = np.maximum(0, int(name[0]) - 1).astype(np.int)
            image_ind = np.maximum(0, int(name[1]) - 1).astype(np.int)
            k_value = np.double(name[2])
        except ValueError as e:
            self.logger.error("Image filename wrong format: {0}".format(name))
            self.result = e
            self.cancel()
            return False
        except IndexError as e:
            self.logger.error("Image filename wrong format: {0}".format(name))
            self.result = e
            self.cancel()
            return False
        filename = os.path.join(self.path, self.image_name)
        image = np.array(Image.open(filename))
        self.result = QuadImage(k_ind, k_value, image_ind, image)


class SaveQuadImageTask(Task):
    def __init__(self, quad_image, save_path=".", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.quad_image = quad_image        # type: QuadImage
        self.save_path = save_path
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
        filename = "{0}_{1}_{2:.5f}_.png".format(self.quad_image.k_ind, self.quad_image.image_ind,
                                                 self.quad_image.k_value)
        full_name = os.path.join(self.save_path, filename)
        with open(full_name, "wb") as fh:
            try:
                write_png(fh, self.quad_image.image, filter_type=1)
            except Exception as e:
                self.logger.error("Image error: {0}".format(e))
                self.logger.error("Image type: {0}".format(type(self.quad_image.image)))
                self.result = e
                self.cancel()
                return
        self.result = True


class LoadQuadScanDirTask(Task):
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
        filename = "daq_info.txt"
        if os.path.isfile(os.path.join(load_dir, filename)) is False:
            e = "daq_info.txt not found in {0}".format(load_dir)
            self.result = e
            self.logger.error(e)
            self.cancel()
            return

        logger.info("{0}: Loading Jason format data".format(self))
        data_dict = dict()
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
        data_dict["roi_center"] = [np.double(rc[0]), np.double(rc[1])]
        rd = data_dict["roi_dim"].split(" ")
        data_dict["roi_dim"] = [np.double(rd[0]), np.double(rd[1])]
        try:
            data_dict["bpp"] = np.int(data_dict["bpp"])
        except KeyError:
            data_dict["bpp"] = 16
            self.logger.debug("{1} Loaded data_dict: \n{0}".format(pprint.pformat(data_dict), self))

        self.acc_params = AcceleratorParameters(electron_energy=float(data_dict["beam_energy"]),
                                                quad_length=float(data_dict["quad_length"]),
                                                quad_screen_dist=float(data_dict["quad_2_screen"]),
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


def process_image_func(image, k_ind, k_value, image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3,
                       bpp=16, normalize=False, enabled=True):
    """
    Function for processing images. To be run in a process pool. ROI, thresholding, median filtering.
    Centroid and second moment extraction. Horizontal and vertical lineouts. Normalize to image bit depth.
    If the image is empty (after filtering), set enabled=False

    :param image: Numpy 2d array with image data
    :param k_ind: Index of the k value in the scan (used for creating ProcessedImage structure)
    :param k_value: Actual k value (used for creating ProcessedImage structure)
    :param image_ind: Index of the image in the scan (used for creating ProcessedImage structure)
    :param threshold: Threshold value under which a pixel is set to 0. Done after normalization, cropping,
                      and median filtering. If None automatic thresholding is done by taking average value of
                      top left and bottom right corners *3.
    :param roi_cent: Center pixel of the ROI, (x,y) tuple
    :param roi_dim: ROI pixel dimensions (w,h) tuple
    :param cal: Pixel calibration in m/pixel
    :param kernel: Median filter kernel size
    :param bpp: Bits per pixel (used for normalization)
    :param normalize: True if normalization is to be performed
    :param enabled: True if this image is enabled (can be overwritten if the image is empty)
    :return: ProcessedImage structure
    """
    # The process function is called from a wrapper function "clearable_pool_worker" that catches exceptions
    # and propagates them to the PoolTask via an output queue. It is therefore not needed to catch
    # exceptions here.
    # logger.debug("Processing image {0}, {1} in pool, size {2}".format(k_ind, image_ind, image.shape))
    print("Processing image {0}, {1} in pool, size {2}".format(k_ind, image_ind, image.shape))
    t0 = time.time()
    # logger.debug("Threshold={0}, cal={1}, kernel={2}".format(threshold, cal, kernel))
    try:
        x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
        y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])

        # Extract ROI and convert to double:
        pic_roi = np.double(image[x[0]:x[1], y[0]:y[1]])
    except IndexError:
        pic_roi = np.double(image)
    # logger.debug("pic_roi size: {0}".format(pic_roi.shape))
    # Normalize pic to 0-1 range, where 1 is saturation:
    n = 2 ** bpp
    # if image.dtype == np.int32:
    #     n = 2 ** 16
    # elif image.dtype == np.uint8:
    #     n = 2 ** 8
    # else:
    #     n = 1

    # logger.debug("Before medfilt, pic roi {0}, kernel {1}".format(pic_roi.shape, kernel))
    # Median filtering:
    try:
        if normalize is True:
            pic_roi = medfilt2d(pic_roi / n, kernel)
        else:
            pic_roi = medfilt2d(pic_roi, kernel)
    except ValueError as e:
        # logger.warning("Medfilt kernel value error: {0}".format(e))
        print("Medfilt kernel value error: {0}".format(e))

    print("Medfilt done")
    # Threshold image
    try:
        if threshold is None:
            threshold = pic_roi[0:20, 0:20].mean()*3 + pic_roi[-20:, -20:].mean()*3
        pic_roi[pic_roi < threshold] = 0.0
    except Exception:
        pic_roi = pic_roi

    line_x = pic_roi.sum(0)
    line_y = pic_roi.sum(1)
    q = line_x.sum()  # Total signal (charge) in the image

    # enabled = False
    l_x_n = np.sum(line_x)
    l_y_n = np.sum(line_y)

    # Enable point only if there is data:
    if l_x_n <= 0.0:
        enabled = False

    try:
        x_v = cal[0] * np.arange(line_x.shape[0])
        y_v = cal[1] * np.arange(line_y.shape[0])
        x_cent = np.sum(x_v * line_x) / l_x_n
        sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
        y_cent = np.sum(y_v * line_y) / l_y_n
        sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
    except Exception as e:
        print(e)
        sigma_x = None
        sigma_y = None
        x_cent = 0
        y_cent = 0

    print("Sigma calculated")

    # Store processed data
    result = ProcessedImage(k_ind=k_ind, k_value=k_value, image_ind=image_ind, pic_roi=pic_roi,
                            line_x=line_x, line_y=line_y, x_cent=x_cent, y_cent=y_cent,
                            sigma_x=sigma_x, sigma_y=sigma_y, q=q, enabled=enabled, threshold=threshold)
    # logger.debug("Image {0}, {1} processed in pool, time {2:.2f} ms".format(k_ind, image_ind, 1e3*(time.time()-t0)))
    return result


class ImageProcessorTask(Task):
    def __init__(self, roi_cent=None, roi_dim=None, threshold=None, cal=[1.0, 1.0], kernel=3, process_exec="process",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self,  name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.kernel = kernel
        self.cal = cal
        self.threshold = threshold
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim
        if process_exec == "process":
            self.processor = ProcessPoolTask(process_image_func, name="process_pool", callback_list=[self._image_done])
        else:
            self.processor = ThreadPoolTask(process_image_func, name="thread_pool", callback_list=[self._image_done])
        # self.processor = ProcessPoolTask(test_f, name="process_pool")
        self.stop_processing_event = threading.Event()

        self.queue_empty_event = threading.Event()
        self.pending_images_in_queue = 0
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
        self.pending_images_in_queue = 0
        self.stop_processing_event.clear()
        if self.processor.is_started() is False:
            self.processor.start()
        self.stop_processing_event.wait(self.timeout)
        if self.stop_processing_event.is_set() is False:
            # Timeout occured
            self.cancel()
            return
        self.logger.info("{0} exit processing".format(self))
        self.processor.finish_processing()
        self.logger.debug("{0}: wait for processpool".format(self))
        self.processor.get_result(wait=True, timeout=self.timeout)
        self.logger.debug("{0}: processpool done".format(self))
        self.processor.clear_callback_list()
        self.processor.stop_processes(terminate=True)
        self.result = True

    def process_image(self, data, bpp=16, enabled=True):
        """
        Send an image for processing. The image can be a QuadImage or a task with a QuadImage as result.

        :param data: QuadImage or task with QuadImage as result
        :param bpp: Bits per pixel in the image
        :param enabled: Should this image be tagged as enabled in the analysis
        :return:
        """
        self.queue_empty_event.clear()
        self.pending_images_in_queue += 1
        if isinstance(data, Task):
            # The task is from a callback
            quad_image = data.get_result(wait=False)    # type: QuadImage
        else:
            quad_image = data                           # type: QuadImage
        self.logger.debug("{0}: Adding image {1} {2} to processing queue".format(self, quad_image.k_ind,
                                                                                 quad_image.image_ind))
        if self.roi_dim is None:
            roi_dim = quad_image.image.shape
        else:
            roi_dim = self.roi_dim
        if self.roi_cent is None:
            roi_cent = [roi_dim[0]/2, roi_dim[1]/2]
        else:
            roi_cent = self.roi_cent
        self.processor.add_work_item(image=quad_image.image, k_ind=quad_image.k_ind, k_value=quad_image.k_value,
                                     image_ind=quad_image.image_ind, threshold=self.threshold, roi_cent=roi_cent,
                                     roi_dim=roi_dim, cal=self.cal, kernel=self.kernel, bpp=bpp, normalize=False,
                                     enabled=enabled)

    def _image_done(self, processor_task):
        # type: (ProcessPoolTask) -> None
        if self.is_done() is False:
            self.logger.debug("{0}: Image processed. {1} images in queue".format(self, self.pending_images_in_queue))
            self.result = processor_task.get_result(wait=False)
            self.pending_images_in_queue -= 1
            if self.pending_images_in_queue <= 0:
                self.queue_empty_event.set()
            if processor_task.is_done() is False:
                self.logger.debug("Calling {0} callbacks".format(len(self.callback_list)))
                for callback in self.callback_list:
                    callback(self)
            else:
                self.logger.debug("{0}: ProcessPoolTask done. Stop processing.".format(self))
                self.stop_processing()

    def stop_processing(self):
        self.logger.debug("{0}: Setting STOP_PROCESSING flag".format(self))
        self.stop_processing_event.set()

    def set_roi(self, roi_cent, roi_dim):
        self.logger.debug("{0}: Setting ROI center {1}, ROI dim {2}".format(self, roi_cent, roi_dim))
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim

    def set_processing_parameters(self, threshold, cal, kernel):
        self.logger.info("{0} Setting threshold={1}, cal={2}, kernel={3}".format(self, threshold, cal, kernel))
        self.threshold = threshold
        self.kernel = kernel
        self.cal = cal

    def wait_for_queue_empty(self):
        self.queue_empty_event.wait(self.timeout)

    def cancel(self):
        self.stop_processing()
        Task.cancel()


var_dict = dict()


def init_worker(sh_mem_image, sh_mem_roi, image_shape):
    var_dict["image"] = sh_mem_image
    var_dict["image_shape"] = image_shape
    var_dict["roi"] = sh_mem_roi


def work_func_shared(mem_ind, im_size, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False):
    t0 = time.time()
    try:
        shape = var_dict["image_shape"]
        # logger.debug("Processing image {0} in pool".format(mem_ind))
        image = np.frombuffer(var_dict["image"], "i", shape[1] * shape[2],
                              shape[1] * shape[2] * mem_ind * np.dtype("i").itemsize).reshape((shape[1], shape[2]))
        roi = np.frombuffer(var_dict["roi"], "f", roi_dim[0] * roi_dim[1],
                            shape[1] * shape[2] * mem_ind * np.dtype("f").itemsize).reshape(roi_dim)
        # logger.debug("{0}: Mem copy time {1:.2f} ms".format(mem_ind, (time.time() - t0) * 1e3))
        try:
            x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
            y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])
            # Extract ROI and convert to double:
            pic_roi = np.float32(image[x[0]:x[1], y[0]:y[1]])
        except IndexError:
            pic_roi = np.float32(image)
        n = 2 ** bpp
        # logger.debug("Pic roi {0}".format(image_ind))

        # Median filtering:
        try:
            if normalize is True:
                pic_roi = medfilt2d(pic_roi / n, kernel)
            else:
                pic_roi = medfilt2d(pic_roi, kernel)
        except ValueError as e:
            logger.warning("{0}======================================".format(mem_ind))
            logger.warning("{1}: Medfilt kernel value error: {0}".format(e, mem_ind))
            # print("Medfilt kernel value error: {0}".format(e))
        # logger.debug("Medfilt {0}".format(image_ind))

        # Threshold image
        try:
            if threshold is None:
                threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
            pic_roi[pic_roi < threshold] = 0.0
        except Exception:
            logger.warning("{0}======================================".format(mem_ind))
            logger.exception("{0}: Could not threshold.".format(mem_ind))
            pic_roi = pic_roi

        # Centroid and sigma calculations:
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        q = line_x.sum()  # Total signal (charge) in the image
        l_x_n = np.sum(line_x)
        l_y_n = np.sum(line_y)
        # Enable point only if there is data:
        if l_x_n <= 0.0:
            enabled = False
        else:
            enabled = True
        try:
            x_v = cal[0] * np.arange(line_x.shape[0])
            y_v = cal[1] * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
        except Exception as e:
            print(e)
            sigma_x = None
            sigma_y = None
            x_cent = 0
            y_cent = 0
        np.copyto(roi, pic_roi)
    except Exception as e:
        logger.warning("{0}======================================".format(mem_ind))
        logger.exception("{0} Work function error".format(mem_ind))
        return mem_ind, e
    # logger.debug("{1}: Process time {0:.2f} ms".format((time.time()-t0) * 1e3, mem_ind))
    return mem_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled


class ImageProcessorTask2(Task):
    def __init__(self, roi_cent=None, roi_dim=None, threshold=None, cal=[1.0, 1.0], kernel=3,
                 image_size=[2000, 2000], number_processes=None,
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self,  name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.kernel = kernel
        self.cal = cal
        self.threshold = threshold
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim

        self.work_func = work_func_shared
        self.lock = threading.Lock()
        self.finish_process_event = threading.Event()
        self.result_ready_event = threading.Event()
        self.queue_empty_event = threading.Event()
        self.pending_work_items = list()

        self.sh_mem_rawarray = None         # Shared memory array
        self.sh_mem_roi_rawarray = None
        self.sh_np_array = None             # Numpy array from the shared buffer. Shape [im_x, im_y, n_proc]
        self.sh_roi_np_array = None
        self.image_size = image_size
        self.pending_image_size = None
        self.update_image_size_proc_id = None
        self.job_queue = Queue.Queue()
        self.current_job_list = None
        self.mem_ind_queue = Queue.Queue()

        if number_processes is None:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = number_processes

        self.pool = None
        self.next_process_id = 0
        self.completed_work_items = 0
        self.id_lock = threading.Lock()

        self.stop_launcher_thread_flag = False

        self.logger.setLevel(logging.INFO)

    def run(self):
        self._create_processes()
        Task.run(self)      # self.action is called here

    def start(self):
        self.next_process_id = 0
        Task.start(self)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))

        self.completed_work_items = 0

        self.finish_process_event.wait(self.timeout)
        if self.finish_process_event.is_set() is False:
            self.cancel()
            return
        self.logger.debug("{0} Finish process event set".format(self))

        t0 = time.time()
        while self.completed_work_items < self.next_process_id:
            time.sleep(0.01)
            if time.time() - t0 > 5.0:
                self.logger.error("{0}: Timeout waiting for {1} work items to complete".format(self, self.next_process_id - self.completed_work_items))
                self._stop_processes(terminate=True)
                # self.result = self.result_dict
                return
        self._stop_processes(terminate=False)
        self.result = True
        self.pool = None

    def process_image(self, data, bpp=16, enabled=True):
        """
        Send an image for processing. The image can be a QuadImage or a task with a QuadImage as result.

        :param data: QuadImage or task with QuadImage as result
        :param bpp: Bits per pixel in the image
        :param enabled: Should this image be tagged as enabled in the analysis
        :return:
        """
        self.queue_empty_event.clear()
        if isinstance(data, Task):
            # The task is from a callback
            quad_image = data.get_result(wait=False)    # type: QuadImage
        else:
            quad_image = data                           # type: QuadImage
        self.logger.debug("{0}: Adding image {1} {2} to processing queue".format(self, quad_image.k_ind,
                                                                                 quad_image.image_ind))
        if self.roi_dim is None:
            roi_dim = quad_image.image.shape
        else:
            roi_dim = self.roi_dim
        if self.roi_cent is None:
            roi_cent = [roi_dim[0]/2, roi_dim[1]/2]
        else:
            roi_cent = self.roi_cent

        # Check if image resize is needed:
        if (quad_image.image.shape[0] * quad_image.image.shape[1]) > (self.image_size[0] * self.image_size[1]):
            self.logger.debug("{0}: Got image size {1}, shared memory size {2}. "
                              "Resize needed.".format(self, quad_image.image.shape, self.image_size))
            # Launch in thread?
            self.set_image_size(quad_image.image.shape)

        self._add_work_item(image=quad_image.image, k_ind=quad_image.k_ind, k_value=quad_image.k_value,
                            image_ind=quad_image.image_ind, threshold=self.threshold, roi_cent=roi_cent,
                            roi_dim=roi_dim, cal=self.cal, kernel=self.kernel, bpp=bpp, normalize=False,
                            enabled=enabled)

    def set_roi(self, roi_cent, roi_dim):
        self.logger.debug("{0}: Setting ROI center {1}, ROI dim {2}".format(self, roi_cent, roi_dim))
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim

    def set_processing_parameters(self, threshold, cal, kernel):
        self.logger.info("{0} Setting threshold={1}, cal={2}, kernel={3}".format(self, threshold, cal, kernel))
        self.threshold = threshold
        self.kernel = kernel
        self.cal = cal

    def set_image_size(self, image_size):
        if image_size == self.pending_image_size:
            self.logger.debug("{0} size change pending".format(self))
            return
        self.logger.info("{0} Setting image size {1}".format(self, image_size))
        with self.lock:
            if self.update_image_size_proc_id is None:
                self.update_image_size_proc_id = self.next_process_id - 1
                self.pending_image_size = image_size
                update_immediate = self.update_image_size_proc_id <= self.completed_work_items
            else:
                update_immediate = False
        self.logger.debug("{0} Completed workitems: {1}, "
                          "updating image size at: {2}".format(self, self.completed_work_items,
                                                               self.update_image_size_proc_id))
        if update_immediate:
            self._create_processes()

    def wait_for_queue_empty(self):
        self.queue_empty_event.wait(self.timeout)

    def cancel(self):
        self.finish_processing()
        Task.cancel(self)

    def _create_processes(self):
        if self.pool is not None:
            self.pool.terminate()
        self.logger.info("{1}: Creating {0} processes".format(self.num_processes, self))
        n_mem = self.num_processes

        with self.lock:
            if self.pending_image_size is not None:
                self.image_size = self.pending_image_size

        self.logger.info("{0} Init shared memory of size {1}x{2}".format(self, n_mem, self.image_size))
        self.sh_mem_rawarray = multiprocessing.RawArray("i", n_mem * self.image_size[0] * self.image_size[1])
        self.sh_np_array = np.frombuffer(self.sh_mem_rawarray, dtype="i").reshape((n_mem,
                                                                                   self.image_size[0],
                                                                                   self.image_size[1]))
        self.sh_mem_roi_rawarray = multiprocessing.RawArray("f", n_mem * self.image_size[0] * self.image_size[1])

        while not self.mem_ind_queue.empty():
            self.mem_ind_queue.get_nowait()
        [self.mem_ind_queue.put(x) for x in range(n_mem)]  # Fill queue with available memory indices
        self.current_job_list = [None for x in range(n_mem)]

        self.pool = multiprocessing.Pool(self.num_processes, initializer=init_worker,
                                         initargs=(self.sh_mem_rawarray, self.sh_mem_roi_rawarray,
                                                   (n_mem, self.image_size[0], self.image_size[1])))
        # time.sleep(0.1)
        with self.lock:
            self.update_image_size_proc_id = None
        self.logger.info("Process creation complete")
        self._job_launch()

    # def _init_shared_memory(self):
    #     n_mem = self.num_processes
    #     with self.lock:
    #         if self.pending_image_size is not None:
    #             self.image_size = self.pending_image_size
    #             self.update_image_size_proc_id = False
    #
    #     self.logger.info("{0} Init shared memory of size {1}x{2}".format(self, n_mem, self.image_size))
    #     self.sh_mem_rawarray = multiprocessing.RawArray("i", self.image_size[0] * self.image_size[1] * n_mem)
    #     self.sh_np_array = np.frombuffer(self.sh_mem_rawarray, dtype="i").reshape((self.image_size[0],
    #                                                                                self.image_size[1],
    #                                                                                n_mem))
    #     self.sh_mem_roi_rawarray = multiprocessing.RawArray("f", self.image_size[0] * self.image_size[1] * n_mem)
    #     while not self.mem_ind_queue.empty():
    #         self.mem_ind_queue.get_nowait()
    #     [self.mem_ind_queue.put(x) for x in range(n_mem)]  # Fill queue with available memory indices
    #     self.current_job_list = [None for x in range(n_mem)]

    def _add_work_item(self, image, k_ind, k_value, image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3,
                       bpp=16, normalize=False, enabled=True):
        # self.logger.debug("{0}: Adding work item".format(self))
        # self.logger.debug("{0}: Args: {1}, kwArgs: {2}".format(self, args, kwargs))
        if not self.finish_process_event.is_set():
            with self.lock:
                proc_id = self.next_process_id
                self.next_process_id += 1
            job = JobStruct(image=image, k_ind=k_ind, k_value=k_value, image_ind=image_ind, threshold=threshold,
                            roi_cent=roi_cent, roi_dim=roi_dim, cal=cal, kernel=kernel, bpp=bpp, normalize=normalize,
                            enabled=enabled, job_proc_id=proc_id)
            self.job_queue.put(job)
            self.logger.debug("{0}: Work item added to queue. Process id: {1}, job queue length: {2}, "
                              "mem_slots {3}".format(self, proc_id, self.job_queue.qsize(), self.mem_ind_queue.qsize()))
            if self.mem_ind_queue.qsize() > 0:
                th = threading.Thread(target=self._job_launch)
                th.start()
                # self._job_launch()
            else:
                self.logger.debug("{0} No available memory slots. "
                                  "Waiting for job completion before starting.".format(self))

    def _job_launch(self):
        with self.lock:
            if self.update_image_size_proc_id is not None:
                self.logger.info("{0} Not launching new jobs while waiting for image size change".format(self))
                # Do not launch new job when waiting for image size change
                return

        try:
            job = self.job_queue.get(False, 0.05)  # type: JobStruct
        except Queue.Empty:
            # If the queue was empty, timeout and check the stop_result_flag again
            self.logger.debug("{0} Job queue empty. Not launching new job. "
                              "Next proc id {1}".format(self, self.next_process_id))
            return

        # wait for memory to be available:
        while not self.stop_launcher_thread_flag:
            try:
                ind = self.mem_ind_queue.get(False, 0.05)
            except Queue.Empty:
                # If the queue was empty, timeout and check the stop_result_flag again
                continue
            self.logger.debug("{0} Job received: {1} {2}".format(self, job.k_ind, job.image_ind))

            roi_d = [int(job.roi_dim[0]), int(job.roi_dim[1])]
            roi_c = [int(job.roi_cent[0]), int(job.roi_cent[1])]

            # copy image data to shared memory:
            im_size = job.image.shape
            np.copyto(self.sh_np_array[ind, 0:im_size[0], 0:im_size[1]], job.image)
            kwargs = {"mem_ind": ind, "im_size": im_size, "threshold": job.threshold, "roi_cent": roi_c,
                      "roi_dim": roi_d, "cal": job.cal, "kernel": job.kernel,
                      "bpp": job.bpp, "normalize": job.normalize}

            # Save job in list:
            self.current_job_list[ind] = job

            # Start processing:
            if not self.stop_launcher_thread_flag:
                self.logger.debug("{0} apply async proc {1}".format(self, ind))
                self.pool.apply_async(self.work_func, kwds=kwargs, callback=self._pool_callback)

            if self.mem_ind_queue.qsize() > 0:
                try:
                    job = self.job_queue.get(False, 0.05)  # type: JobStruct
                except Queue.Empty:
                    return
            else:
                return

    def _pool_callback(self, result):
        self.logger.debug("Pool callback returned {0}".format(result[0]))
        self.completed_work_items += 1

        if self.is_done() is False:
            ind = result[0]
            if isinstance(result[1], Exception):
                self.logger.exception("{0} pool callback exception: {1}".format(self, result))
                self.result = result[1]
            else:
                x_cent = result[1]
                sigma_x = result[2]
                y_cent = result[3]
                sigma_y = result[4]
                q = result[5]
                enabled = result[6]
                job = self.current_job_list[ind]    # type: JobStruct
                if not job.enabled:
                    enabled = False

                roi_d = [int(job.roi_dim[0]), int(job.roi_dim[1])]
                pic_roi = np.frombuffer(self.sh_mem_roi_rawarray, "f", roi_d[0] * roi_d[1],
                                        self.image_size[0] * self.image_size[1] *
                                        ind * np.dtype("f").itemsize).reshape(roi_d).copy()
                line_x = pic_roi.sum(0)
                line_y = pic_roi.sum(1)
                self.result = ProcessedImage(k_ind=job.k_ind, k_value=job.k_value, image_ind=job.image_ind,
                                             pic_roi=pic_roi, line_x=line_x, line_y=line_y, x_cent=x_cent,
                                             sigma_x=sigma_x, y_cent=y_cent, sigma_y=sigma_y, q=q, enabled=enabled,
                                             threshold=job.threshold)
                for callback in self.callback_list:
                    callback(self)

            # Mark this index as free for processing:
            self.mem_ind_queue.put(ind)
            # Check if it is time to change image size:
            with self.lock:
                if self.update_image_size_proc_id is not None:
                    update = self.update_image_size_proc_id <= self.completed_work_items
                else:
                    update = False
            if update:
                self._create_processes()

            self._job_launch()

            pending_images_in_queue = self.next_process_id - self.completed_work_items
            if pending_images_in_queue <= 0:
                self.queue_empty_event.set()

            # Signal that a result is ready. This will trigger a get_result(wait=True).
            self.result_ready_event.set()
            self.result_ready_event.clear()

    def _stop_processes(self, terminate=True):
        self.logger.info("{0}: Stopping processes".format(self))
        self.stop_launcher_thread_flag = True
        try:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
        except AttributeError:
            pass

        self.logger.info("{0}: Processes stopped".format(self))

    def finish_processing(self):
        self.finish_process_event.set()

    def get_result(self, wait, timeout=-1):
        if self.completed is not True:
            if wait is True:
                if timeout > 0:
                    self.result_ready_event.wait(timeout)
                else:
                    self.result_ready_event.wait()
        return self.result

    def get_remaining_number_images(self):
        return self.next_process_id-self.completed_work_items

    def emit(self):
        self.result_ready_event.set()
        Task.emit(self)


class ProcessAllImagesTask(Task):
    def __init__(self, quad_scan_data, threshold=None, kernel_size=3,
                 image_processor_task=None, process_exec_type="process",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.INFO)
        self.quad_scan_data = quad_scan_data    # type: QuadScanData
        self.threshold = threshold
        self.kernel = kernel_size
        self.images_done_event = threading.Event()
        self.pending_images = 0

        if image_processor_task is None:
            self.image_processor = ImageProcessorTask(threshold=threshold, kernel=kernel_size,
                                                      process_exec=process_exec_type,
                                                      trigger_dict=trigger_dict, name="processall_image_proc")
        else:
            # If the image_processor was supplied, don't add self as trigger.
            self.image_processor = image_processor_task     # Type: ImageProcessorTask
        if self.image_processor.is_started() is False:
            self.logger.info("{0} Image processor not started. Starting. "
                             "Cancel status: {1}".format(self, self.image_processor.is_cancelled()))
            self.image_processor.start()
        self.image_processor.add_callback(self.processed_image_done)
        self.processed_image_list = list()

    def action(self):
        self.logger.info("{0}: entering action".format(self))

        acc_params = self.quad_scan_data.acc_params
        n_images = acc_params.num_k * acc_params.num_images
        pic_roi = np.zeros((int(acc_params.roi_dim[0]), int(acc_params.roi_dim[1])), dtype=np.float32)
        [self.processed_image_list.append(pic_roi) for x in range(n_images)]

        self.image_processor.set_roi(self.quad_scan_data.acc_params.roi_center, self.quad_scan_data.acc_params.roi_dim)
        self.image_processor.set_processing_parameters(self.threshold, self.quad_scan_data.acc_params.cal, self.kernel)
        # self.image_processor.start()
        self.pending_images = len(self.quad_scan_data.images)
        for ind, image in enumerate(self.quad_scan_data.images):
            try:
                en = self.quad_scan_data.proc_images[ind].enabled
            except IndexError:
                en = True
            except AttributeError:
                en = True
            self.image_processor.process_image(image, enabled=en)
        self.logger.debug("{0}: Starting wait for images".format(self))
        self.images_done_event.wait(self.timeout)
        # self.image_processor.stop_processing()
        # self.image_processor.get_result(wait=True)      # Wait for all images to finish processing
        self.logger.debug("{0}: Storing result".format(self))
        self.result = self.processed_image_list
        self.image_processor.clear_callback_list()

    def processed_image_done(self, image_processor_task):
        # type: (ImageProcessorTask) -> None
        proc_image = image_processor_task.get_result(wait=False)    # type: ProcessedImage
        if image_processor_task.is_done() is False:
            with self.lock:
                ind = proc_image.k_ind * self.quad_scan_data.acc_params.num_images + proc_image.image_ind
                self.logger.debug(
                    "{0} Adding processed image {1} {2} to list at index {3}".format(self,
                                                                                     proc_image.k_ind,
                                                                                     proc_image.image_ind,
                                                                                     ind))
                self.processed_image_list[ind] = proc_image
                # self.processed_image_list.append(proc_image)
                self.pending_images -= 1
                if self.pending_images <= 0:
                    self.images_done_event.set()
        else:
            self.images_done_event.set()

    def cancel(self):
        # self.image_processor.cancel()
        Task.cancel(self)


def work_func_shared2(mem_ind, im_ind, im_size, threshold, roi_cent, roi_dim,
                     cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False, keep_charge_ratio=1.0):
    t0 = time.time()
    try:
        shape = var_dict["image_shape"]
        # logger.debug("Processing image {0} in pool".format(mem_ind))
        image = np.frombuffer(var_dict["image"], "i", shape[1] * shape[2],
                              shape[1] * shape[2] * mem_ind * np.dtype("i").itemsize).reshape((shape[1], shape[2]))
        roi = np.frombuffer(var_dict["roi"], "f", roi_dim[0] * roi_dim[1],
                            shape[1] * shape[2] * mem_ind * np.dtype("f").itemsize).reshape(roi_dim)
        t1 = time.time()
        # logger.debug("{0}: Mem copy time {1:.2f} ms".format(mem_ind, (t1 - t0) * 1e3))
        try:
            x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
            y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])
            # Extract ROI and convert to double:
            pic_roi = np.float32(image[x[0]:x[1], y[0]:y[1]])
        except IndexError:
            pic_roi = np.float32(image)
        n = 2 ** bpp
        # t2 = time.time()
        # logger.debug("{0}: Pic roi time {1:.2f} ms".format(im_ind, (t2-t1)*1e3))

        # Median filtering:
        try:
            if normalize is True:
                pic_roi = medfilt2d(pic_roi / n, kernel)
            else:
                pic_roi = medfilt2d(pic_roi, kernel)
        except ValueError as e:
            logger.warning("{0}======================================".format(mem_ind))
            logger.warning("{1}: Medfilt kernel value error: {0}".format(e, mem_ind))
        # logger.debug("Medfilt {0}".format(image_ind))
        # t3 = time.time()
        # logger.debug("{0}: Medfilt time {1:.2f} ms".format(im_ind, (t3-t2)*1e3))

        # Threshold image
        try:
            if threshold is None:
                threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
            pic_roi[pic_roi < threshold] = 0.0
        except Exception:
            logger.warning("{0}======================================".format(mem_ind))
            logger.exception("{0}: Could not threshold.".format(mem_ind))
            pic_roi = pic_roi

        # t4 = time.time()
        # logger.debug("{0}: Threshold time {1:.2f} ms".format(im_ind, (t4-t3)*1e3))

        # Filter out charge:
        if normalize:
            n_bins = np.unique(pic_roi.flatten()).shape[0]
        else:
            n_bins = int(pic_roi.max())
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(pic_roi, n_bins)
        hq = (h[0]*h[1][:-1]).cumsum()
        hq = hq / np.float(hq.max())
        # hq = (hq.astype(np.float) / hq.max()).cumsum()
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        d = (h[1][1] - h[1][0])/2.0
        th_q = h[1][th_ind] - d
        pic_roi[pic_roi < th_q] = 0.0
        # logger.debug("Pic_roi max: {0}, threshold index: {1}, threshold: {2}, ch ratio: {3}\n"
        #              "hq: {4}".format(n_bins, th_ind, th_q, keep_charge_ratio, hq[0:20]))

        # Centroid and sigma calculations:
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        q = line_x.sum()  # Total signal (charge) in the image
        l_x_n = np.sum(line_x)
        l_y_n = np.sum(line_y)
        # Enable point only if there is data:
        if l_x_n <= 0.0:
            enabled = False
        else:
            enabled = True
        try:
            x_v = cal[0] * np.arange(line_x.shape[0])
            y_v = cal[1] * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
        except Exception as e:
            print(e)
            sigma_x = None
            sigma_y = None
            x_cent = 0
            y_cent = 0

        # t5 = time.time()
        # logger.debug("{0}: Sigma time {1:.2f} ms".format(im_ind, (t5-t4)*1e3))

        np.copyto(roi, pic_roi)
    except Exception as e:
        logger.warning("{0}======================================".format(mem_ind))
        logger.exception("{0} Work function error".format(mem_ind))
        return mem_ind, e
    # logger.debug("{1}: Process time {0:.2f} ms".format((time.time()-t0) * 1e3, mem_ind))
    return mem_ind, im_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled


def work_func_shared_cv2(mem_ind, im_ind, im_size, threshold, roi_cent, roi_dim,
                     cal=[1.0, 1.0], kernel=3, bpp=12, normalize=False, keep_charge_ratio=1.0):
    t0 = time.time()
    try:
        shape = var_dict["image_shape"]
        # logger.debug("Processing image {0} in pool".format(mem_ind))
        image = np.frombuffer(var_dict["image"], "i", shape[1] * shape[2],
                              shape[1] * shape[2] * mem_ind * np.dtype("i").itemsize).reshape((shape[1], shape[2]))
        roi = np.frombuffer(var_dict["roi"], "f", roi_dim[0] * roi_dim[1],
                            shape[1] * shape[2] * mem_ind * np.dtype("f").itemsize).reshape(roi_dim)
        t1 = time.time()
        # logger.debug("{0}: Mem copy time {1:.2f} ms".format(mem_ind, (t1 - t0) * 1e3))
        try:
            x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
            y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])
            # Extract ROI and convert to double:
            pic_roi = np.float32(image[x[0]:x[1], y[0]:y[1]])
        except IndexError:
            pic_roi = np.float32(image)
        n = 2 ** bpp
        # t2 = time.time()
        # logger.debug("{0}: Pic roi time {1:.2f} ms".format(im_ind, (t2-t1)*1e3))

        # Median filtering:
        try:
            pic_roi = cv2.medianBlur(pic_roi, kernel)
        except ValueError as e:
            logger.warning("{0}======================================".format(mem_ind))
            logger.warning("{1}: Medfilt kernel value error: {0}".format(e, mem_ind))

        if normalize is True:
            pic_roi = pic_roi / n

        # Threshold image
        try:
            if threshold is None:
                threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
            pic_roi[pic_roi < threshold] = 0.0
        except Exception:
            logger.warning("{0}======================================".format(mem_ind))
            logger.exception("{0}: Could not threshold.".format(mem_ind))
            pic_roi = pic_roi

        # t4 = time.time()
        # logger.debug("{0}: Threshold time {1:.2f} ms".format(im_ind, (t4-t3)*1e3))

        # Filter out charge:
        if normalize:
            n_bins = np.unique(pic_roi.flatten()).shape[0]
        else:
            n_bins = int(pic_roi.max())
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(pic_roi, n_bins)
        hq = (h[0]*h[1][:-1]).cumsum()
        hq = hq / np.float(hq.max())
        # hq = (hq.astype(np.float) / hq.max()).cumsum()
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        d = (h[1][1] - h[1][0])/2.0
        th_q = h[1][th_ind] - d
        pic_roi[pic_roi < th_q] = 0.0
        # logger.debug("Pic_roi max: {0}, threshold index: {1}, threshold: {2}, ch ratio: {3}\n"
        #              "hq: {4}".format(n_bins, th_ind, th_q, keep_charge_ratio, hq[0:20]))

        # Centroid and sigma calculations:
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        q = line_x.sum()  # Total signal (charge) in the image
        l_x_n = np.sum(line_x)
        l_y_n = np.sum(line_y)
        # Enable point only if there is data:
        if l_x_n <= 0.0:
            enabled = False
        else:
            enabled = True
        try:
            x_v = cal[0] * np.arange(line_x.shape[0])
            y_v = cal[1] * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
        except Exception as e:
            print(e)
            sigma_x = None
            sigma_y = None
            x_cent = 0
            y_cent = 0

        # t5 = time.time()
        # logger.debug("{0}: Sigma time {1:.2f} ms".format(im_ind, (t5-t4)*1e3))

        np.copyto(roi, pic_roi)
    except Exception as e:
        logger.warning("{0}======================================".format(mem_ind))
        logger.exception("{0} Work function error".format(mem_ind))
        return mem_ind, e
    # logger.debug("{1}: Process time {0:.2f} ms".format((time.time()-t0) * 1e3, mem_ind))
    return mem_ind, im_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled


def work_func_shared_cv2_mask(mem_ind, im_ind, im_size, threshold, roi_cent, roi_dim,
                              cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False, keep_charge_ratio=1.0):
    """

    :param mem_ind: Image index into the shared memory
    :type mem_ind: int
    :param im_ind:
    :type im_ind:
    :param im_size:
    :type im_size:
    :param threshold: Background subtraction threshold. None if automatic thresholding
    :type threshold:
    :param roi_cent: Center pixel of ROI, x,y tuple
    :type roi_cent:
    :param roi_dim: ROI dimensions, width, height tuple
    :type roi_dim:
    :param cal: Pixel size calibration tuple
    :type cal:
    :param kernel: Median filter kernel size
    :type kernel: Odd int
    :param bpp: Bits per pixel
    :type bpp:
    :param normalize: True if the image should be normalized to [0:1]
    :type normalize:
    :param keep_charge_ratio: Amount of charge to keep after thresholding
    :type keep_charge_ratio: float [0:1]
    :return:
    :rtype:
    """
    t0 = time.time()
    mask_kern = 5      # Vertical median filter size for mask creation
    try:
        shape = var_dict["image_shape"]
        # logger.debug("Processing image {0} in pool".format(mem_ind))
        image = np.frombuffer(var_dict["image"], "i", shape[1] * shape[2],
                              shape[1] * shape[2] * mem_ind * np.dtype("i").itemsize).reshape((shape[1], shape[2]))
        roi = np.frombuffer(var_dict["roi"], "f", roi_dim[0] * roi_dim[1],
                            shape[1] * shape[2] * mem_ind * np.dtype("f").itemsize).reshape(roi_dim)
        t1 = time.time()
        # logger.debug("{0}: Mem copy time {1:.2f} ms".format(mem_ind, (t1 - t0) * 1e3))
        try:
            x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
            y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])
            # Extract ROI and convert to double:
            pic_roi = np.float32(image[x[0]:x[1], y[0]:y[1]])
        except IndexError:
            pic_roi = np.float32(image)
        n = 2 ** bpp

        # Median filtering:
        try:
            pic_roi = cv2.medianBlur(pic_roi, kernel)
        except ValueError as e:
            logger.warning("{0}======================================".format(mem_ind))
            logger.warning("{1}: Medfilt kernel value error: {0}".format(e, mem_ind))

        if normalize is True:
            pic_roi = pic_roi / n

        if threshold is None:
            # Background level from first 20 columns, one level for each row (the background structure is banded):
            if y[0] > 20:
                pic_bkg = np.float32(image[x[0]:x[1], y[0]-20:y[0]])
                bkg_level = cv2.medianBlur(pic_bkg, kernel).mean(1)
                if normalize:
                    bkg_level /= n
            elif y[1] < image.shape[1] - 20:
                pic_bkg = np.float32(image[x[0]:x[1], y[1]:y[1] + 20])
                bkg_level = cv2.medianBlur(pic_bkg, kernel).mean(1)
                if normalize:
                    bkg_level /= n
            else:
                bkg_level = pic_roi[:, 0:20].mean(1)
            logger.debug("{0:.1f}".format(bkg_level.max()))
            bkg_cut = bkg_level.max() + bkg_level.std() * 3
            logger.debug("Bkg level: {0:.1f}, bkg_cut: {1:.1f}".format(bkg_level.max(), bkg_cut))
        else:
            bkg_cut = threshold

        pic_proc2 = cv2.threshold(pic_roi - (bkg_cut), thresh=0, maxval=1, type=cv2.THRESH_TOZERO)[1]
        # Create mask around the signal spot by heavily median filtering in the vertical direction (mask_kern ~ 25)
        # mask = cv2.threshold(cv2.boxFilter(pic_roi, -1, ksize=(np.maximum(kernel, 7), mask_kern)),
        #                      thresh=bkg_cut, maxval=1, type=cv2.THRESH_BINARY)[1]
        mask = cv2.threshold(cv2.boxFilter(cv2.medianBlur(pic_roi, ksize=5), -1, ksize=(mask_kern, mask_kern)),
                             thresh=bkg_cut, maxval=1, type=cv2.THRESH_BINARY)[1]
        pic_proc3 = cv2.multiply(pic_proc2, mask)

        logger.debug("pic_roi max: {0}, mask sum: {1}, pic_proc3 max: {2}".format(pic_roi.max(), mask.sum(), pic_proc3.max()))

        # t4 = time.time()
        # logger.debug("{0}: Threshold time {1:.2f} ms".format(im_ind, (t4-t3)*1e3))

        # Filter out charge:
        if normalize:
            n_bins = np.unique(pic_proc3.flatten()).shape[0]
        else:
            n_bins = int(pic_proc3.max())
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(pic_proc3, n_bins)
        hq = (h[0]*h[1][:-1]).cumsum()
        hq = hq / np.float(hq.max())
        # hq = (hq.astype(np.float) / hq.max()).cumsum()
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        d = (h[1][1] - h[1][0])/2.0
        th_q = h[1][th_ind] - d
        pic_proc3[pic_proc3 < th_q] = 0.0
        logger.debug("Pic_roi max: {0}, threshold index: {1}, threshold: {2}, ch ratio: {3}\n"
                     "hq: {4}".format(n_bins, th_ind, th_q, keep_charge_ratio, hq[0:20]))

        # Centroid and sigma calculations:
        line_x = pic_proc3.sum(0)
        line_y = pic_proc3.sum(1)
        q = line_x.sum()  # Total signal (charge) in the image
        l_x_n = np.sum(line_x)
        l_y_n = np.sum(line_y)
        # Enable point only if there is data:
        if l_x_n <= 0.0:
            enabled = False
        else:
            enabled = True
        try:
            logger.debug("cal {0}".format(cal))
            try:
                x_v = cal[0] * np.arange(line_x.shape[0])
                y_v = cal[1] * np.arange(line_y.shape[0])
            except TypeError:
                x_v = cal * np.arange(line_x.shape[0])
                y_v = cal * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
        except Exception as e:
            print(e)
            sigma_x = None
            sigma_y = None
            x_cent = 0
            y_cent = 0

        # t5 = time.time()
        # logger.debug("{0}: Sigma time {1:.2f} ms".format(im_ind, (t5-t4)*1e3))

        np.copyto(roi, pic_proc3)
    except Exception as e:
        logger.warning("{0}======================================".format(mem_ind))
        logger.exception("{0} Work function error".format(mem_ind))
        return mem_ind, e
    # logger.debug("{1}: Process time {0:.2f} ms".format((time.time()-t0) * 1e3, mem_ind))
    return mem_ind, im_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled


def work_func_local2(image, im_ind, threshold, roi_cent, roi_dim,
                     cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False, keep_charge_ratio=1.0):
    t0 = time.time()
    try:
        try:
            x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
            y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])
            # Extract ROI and convert to double:
            pic_roi = np.float32(image[x[0]:x[1], y[0]:y[1]])
        except IndexError:
            pic_roi = np.float32(image)
        n = 2 ** bpp

        # Median filtering:
        try:
            if normalize is True:
                pic_roi = medfilt2d(pic_roi / n, kernel)
            else:
                pic_roi = medfilt2d(pic_roi, kernel)
        except ValueError as e:
            logger.warning("{0}======================================".format(im_ind))
            logger.warning("{1}: Medfilt kernel value error: {0}".format(e, im_ind))
        # logger.debug("Medfilt {0}".format(image_ind))
        # t3 = time.time()
        # logger.debug("{0}: Medfilt time {1:.2f} ms".format(im_ind, (t3-t2)*1e3))

        # Threshold image
        try:
            if threshold is None:
                threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
            pic_roi[pic_roi < threshold] = 0.0
        except Exception:
            logger.warning("{0}======================================".format(im_ind))
            logger.exception("{0}: Could not threshold.".format(im_ind))
            pic_roi = pic_roi

        # t4 = time.time()
        # logger.debug("{0}: Threshold time {1:.2f} ms".format(im_ind, (t4-t3)*1e3))

        # Filter out charge:
        if normalize:
            n_bins = np.unique(pic_roi.flatten()).shape[0]
            # Number of bins in histogram: number of unique pixel values
        else:
            n_bins = int(pic_roi.max())
        if n_bins < 1:
            n_bins = 1
        h = np.histogram(pic_roi, n_bins)
        hq = (h[0]*h[1][:-1]).cumsum()      # Scale number of pixels with pixel value to get total charge for this value
        hq = hq / np.float(hq.max())
        th_ind = np.searchsorted(hq, 1-keep_charge_ratio)
        # d = (h[1][1] - h[1][0])/2.0
        d = 0
        th_q = h[1][th_ind] - d
        pic_roi[pic_roi < th_q] = 0.0
        # logger.debug("Pic_roi max: {0}, threshold index: {1}, threshold: {2}, ch ratio: {3}\n"
        #              "hq: {4}".format(n_bins, th_ind, th_q, keep_charge_ratio, hq[0:20]))

        # Centroid and sigma calculations:
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        q = line_x.sum()  # Total signal (charge) in the image
        l_x_n = np.sum(line_x)
        l_y_n = np.sum(line_y)
        # Enable point only if there is data:
        if l_x_n <= 0.0:
            enabled = False
        else:
            enabled = True
        try:
            x_v = cal[0] * np.arange(line_x.shape[0])
            y_v = cal[1] * np.arange(line_y.shape[0])
            x_cent = np.sum(x_v * line_x) / l_x_n
            sigma_x = np.sqrt(np.sum((x_v - x_cent) ** 2 * line_x) / l_x_n)
            y_cent = np.sum(y_v * line_y) / l_y_n
            sigma_y = np.sqrt(np.sum((y_v - y_cent) ** 2 * line_y) / l_y_n)
        except Exception as e:
            print(e)
            sigma_x = None
            sigma_y = None
            x_cent = 0
            y_cent = 0

        # t5 = time.time()
        # logger.debug("{0}: Sigma time {1:.2f} ms".format(im_ind, (t5-t4)*1e3))

    except Exception as e:
        logger.warning("{0}======================================".format(im_ind))
        logger.exception("{0} Work function error".format(im_ind))
        return im_ind, e
    # logger.debug("{1}: Process time {0:.2f} ms".format((time.time()-t0) * 1e3, mem_ind))
    return pic_roi, im_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled


class ProcessAllImagesTask2(Task):
    def __init__(self, image_size=[2000, 2000], num_processes=None, watchdog_time=3.0,
                 process_exec_type="thread", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        """
        The idea is here to process all images in batches in a process pool.

        :param quad_scan_data:
        :param threshold:
        :param kernel_size:
        :param image_processor_task:
        :param process_exec_type:
        :param name:
        :param timeout:
        :param trigger_dict:
        :param callback_list:
        """
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.INFO)

        self.work_func = work_func_shared_cv2_mask

        self.image_size = image_size
        self.quad_scan_data = None    # type: QuadScanData
        self.threshold = None
        self.kernel = None
        self.normalize = False
        self.bpp = 16
        self.enabled_list = None
        self.keep_charge_ratio = None

        self.finish_process_event = threading.Event()
        self.result_done_event = threading.Event()
        self.pending_images = 0
        self.pending_data = None
        self.watchdog_time = watchdog_time
        self.watchdog_timer = threading.Timer(self.watchdog_time, self.restart_processes)
        self.watchdog_timer.cancel()

        self.processed_image_list = list()

        if num_processes is None:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = num_processes
        self.pool = None

        self.sh_mem_rawarray = None
        self.sh_mem_roi_rawarray = None
        self.sh_np_array = None
        self.job_queue = Queue.Queue()
        self.current_jobs_in_process_list = None
        self.mem_ind_queue = Queue.Queue()
        self.next_process_id = None

        self.job_thread = None
        self.start_time = time.time()
        self.prepare_time = time.time()
        self.tot_proc_time = 0.0

    def start(self):
        self.next_process_id = 0
        self.create_pool()
        Task.start(self)

    def action(self):
        self.logger.info("{0}: entering action".format(self))
        self.result_done_event.set()

        self.finish_process_event.wait(self.timeout)
        if self.finish_process_event.is_set() is False:
            self.cancel()
            return
        self.logger.debug("{0} Finish process event set".format(self))

        t0 = time.time()
        while self.pending_images > 0:
            time.sleep(0.01)
            if time.time() - t0 > 5.0:
                self.logger.error("{0}: Timeout waiting for {1} work items to complete".format(self,
                                                                                               self.pending_images))
                self.stop_processes(terminate=True)
                # self.result = self.result_dict
                return
        self.stop_processes(terminate=False)
        self.result = True
        self.pool = None

    def process_images(self, quad_scan_data, threshold, kernel=3, bpp=16, normalize=False,
                       enabled_list=None, keep_charge_ratio=1.0):
        self.logger.info("{0}: New image set {1} images".format(self, len(quad_scan_data.images)))
        self.start_time = time.time()

        # Overwrite pending data if there already was some:
        with self.lock:
            self.pending_data = quad_scan_data
            self.threshold = threshold
            self.kernel = kernel
            self.normalize = normalize
            self.bpp = bpp
            self.enabled_list = enabled_list
            self.keep_charge_ratio = keep_charge_ratio
            self.watchdog_timer.cancel()
            self.watchdog_timer = threading.Timer(self.watchdog_time, self.restart_processes)
            self.watchdog_timer.start()

        # Check if processing is on-going. If not, result done is set. Then we can start immediately
        if self.result_done_event.is_set():
            self.result_done_event.clear()
            self.job_thread = threading.Thread(target=self.prepare_data)
            self.job_thread.start()

    def prepare_data(self):
        with self.lock:
            # Do this again to make sure:
            if not self.result_done_event.is_set():
                self.result_done_event.clear()
            # Transfer pending data to quad_scan_data
            self.quad_scan_data = self.pending_data
            self.pending_data = None

        acc_params = self.quad_scan_data.acc_params
        n_images = acc_params.num_k * acc_params.num_images

        # Prepare processed images as zero-filled images:
        self.processed_image_list = list()
        pic_roi = np.zeros((int(acc_params.roi_dim[0]), int(acc_params.roi_dim[1])), dtype=np.float32)
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        for quad_image in self.quad_scan_data.images:
            proc_image = ProcessedImage(k_ind=quad_image.k_ind,
                                        k_value=quad_image.k_value,
                                        image_ind=quad_image.image_ind,
                                        pic_roi=pic_roi,
                                        line_x=line_x,
                                        line_y=line_y,
                                        x_cent=pic_roi.shape[0]/2,
                                        y_cent=pic_roi.shape[1]/2,
                                        sigma_x=0.0, sigma_y=0.0,
                                        q=0.0, enabled=False, threshold=self.threshold)
            self.processed_image_list.append(proc_image)

        self.pending_images = len(self.quad_scan_data.images)

        # Check image size and see if we need to resize and create new process pool
        quad_image = self.quad_scan_data.images[0]
        if (quad_image.image.shape[0] * quad_image.image.shape[1]) > (self.image_size[0] * self.image_size[1]):
            self.logger.debug("{0}: Got image size {1}, shared memory size {2}. "
                              "Resize needed.".format(self, quad_image.image.shape, self.image_size))
            self.image_size = quad_image.image.shape
            self.create_pool()

        # Fill up job queue
        [self.job_queue.put(ind) for ind in range(len(self.quad_scan_data.images))]

        self.prepare_time = time.time()

        self.job_launcher()
        self.logger.debug("{0}: EXITING JOB THREAD NOW".format(self))

    def job_launcher(self):
        if self.finish_process_event.is_set():
            self.logger.debug("Finish process set. Not launching new job")
            return

        try:
            im_ind = self.job_queue.get(False, 0.05)  # type: int
        except Queue.Empty:
            # If the queue was empty, timeout and check the stop_result_flag again
            self.logger.debug("{0} Job queue empty. Not launching new job. "
                              "Pending images {1}".format(self, self.pending_images))
            return

        # wait for memory to be available:
        while not self.finish_process_event.is_set():
            try:
                ind = self.mem_ind_queue.get(False, 0.05)
            except Queue.Empty:
                # If the queue was empty, timeout and check the stop_result_flag again
                continue
            quad_image = self.quad_scan_data.images[im_ind]
            self.logger.debug("{0} Job {3} received: {1} {2}".format(self,
                                                                     quad_image.k_ind,
                                                                     quad_image.image_ind,
                                                                     im_ind))

            acc_params = self.quad_scan_data.acc_params     # type: AcceleratorParameters
            roi_d = [int(acc_params.roi_dim[0]), int(acc_params.roi_dim[1])]
            roi_c = [int(acc_params.roi_center[0]), int(acc_params.roi_center[1])]

            # copy image data to shared memory:
            im_size = quad_image.image.shape
            if quad_image.image.dtype != np.int:
                np.copyto(self.sh_np_array[ind, 0:im_size[0], 0:im_size[1]], np.int32(quad_image.image))
            else:
                np.copyto(self.sh_np_array[ind, 0:im_size[0], 0:im_size[1]], quad_image.image)
            kwargs = {"mem_ind": ind, "im_ind": im_ind, "im_size": im_size, "threshold": self.threshold,
                      "roi_cent": roi_c, "roi_dim": roi_d, "cal": acc_params.cal, "kernel": self.kernel,
                      "bpp": self.bpp, "normalize": self.normalize, "keep_charge_ratio": self.keep_charge_ratio}

            # Start processing:
            self.logger.debug("{0} apply async proc {1}".format(self, ind))
            with self.lock:
                self.pool.apply_async(self.work_func, kwds=kwargs, callback=self.pool_callback)

            if self.mem_ind_queue.qsize() > 0:
                try:
                    im_ind = self.job_queue.get(False, 0.05)
                except Queue.Empty:
                    return
            else:
                return

    def pool_callback(self, result):
        if self.is_done() is False:
            ind = result[0]
            if isinstance(result[1], Exception):
                self.logger.exception("{0} pool callback exception: {1}".format(self, result))
                # self.result = result[1]
            else:
                im_ind =result[1]
                x_cent = result[2]
                sigma_x = result[3]
                y_cent = result[4]
                sigma_y = result[5]
                q = result[6]
                enabled = result[7]

                quad_image = self.quad_scan_data.images[im_ind]

                try:
                    if self.enabled_list is None:
                        if not self.quad_scan_data.proc_images[im_ind].enabled:
                            enabled = False
                    else:
                        if not self.enabled_list[im_ind]:
                            enabled = False
                except IndexError:
                    self.logger.warning("{0} Enabled list index error".format(self))
                    enabled = False

                acc_params = self.quad_scan_data.acc_params     # type: AcceleratorParameters
                roi_d = [int(acc_params.roi_dim[0]), int(acc_params.roi_dim[1])]
                pic_roi = np.frombuffer(self.sh_mem_roi_rawarray, "f", roi_d[0] * roi_d[1],
                                        self.image_size[0] * self.image_size[1] *
                                        ind * np.dtype("f").itemsize).reshape(roi_d).copy()
                line_x = pic_roi.sum(0)
                line_y = pic_roi.sum(1)
                proc_image = ProcessedImage(k_ind=quad_image.k_ind, k_value=quad_image.k_value,
                                            image_ind=quad_image.image_ind, pic_roi=pic_roi,
                                            line_x=line_x, line_y=line_y, x_cent=x_cent,
                                            sigma_x=sigma_x, y_cent=y_cent, sigma_y=sigma_y,
                                            q=q, enabled=enabled, threshold=self.threshold)
                self.processed_image_list[im_ind] = proc_image
                self.logger.debug("\n==============================================================================="
                                  "\n{0} "
                                  "\nAdding processed image {1} {2} to list at index {3}."
                                  "\nCharge={4}, x_cent={5}"
                                  "\n==============================================================================="
                                  "\n\n".format(self,
                                              proc_image.k_ind,
                                              proc_image.image_ind,
                                              im_ind, proc_image.q, proc_image.x_cent))

            # Mark this index as free for processing:
            self.mem_ind_queue.put(ind)

            self.pending_images -= 1
            if self.pending_images <= 0:
                self.processing_done()
                # self.result_done_event.set()
            else:
                self.job_launcher()

    def processing_done(self):
        tot_time = time.time()-self.start_time
        self.logger.info("\n"
                         "---------------------------------------------------\n"
                         "{0}: \n"
                         "Finished processing images. \n"
                         "    Prepare time: {1:.2f} ms\n"
                         "    Total time: {2:.2f} ms\n"
                         "---------------------------------------------------\n"
                         "".format(self, (self.prepare_time-self.start_time)*1e3, (tot_time)*1e3))
        self.watchdog_timer.cancel()
        self.result_done_event.set()
        with self.lock:
            # self.quad_scan_data = self.quad_scan_data._replace(proc_images=self.processed_image_list)
            # self.result = self.quad_scan_data
            self.result = self.processed_image_list
            self.tot_proc_time = tot_time
        for callback in self.callback_list:
            callback(self)

    def cancel(self):
        # self.image_processor.cancel()
        while True:
            try:
                self.job_queue.get_nowait()
            except Queue.Empty:
                break
        Task.cancel(self)

    def create_pool(self):
        if self.pool is not None:
            self.pool.terminate()
        self.logger.info("{1}: Creating pool with {0} processes".format(self.num_processes, self))

        n_mem = self.num_processes
        self.current_jobs_in_process_list = list()

        n_mem = self.num_processes
        self.logger.info("{0} Init shared memory of size {1}x{2}".format(self, n_mem, self.image_size))
        self.sh_mem_rawarray = multiprocessing.RawArray("i", n_mem * self.image_size[0] * self.image_size[1])
        self.sh_np_array = np.frombuffer(self.sh_mem_rawarray, dtype="i").reshape((n_mem,
                                                                                   self.image_size[0],
                                                                                   self.image_size[1]))
        self.sh_mem_roi_rawarray = multiprocessing.RawArray("f", n_mem * self.image_size[0] * self.image_size[1])

        # Prepare queue for shared mem access. Empty old queue, then put all mem indices as available.
        while not self.mem_ind_queue.empty():
            self.mem_ind_queue.get_nowait()
        [self.mem_ind_queue.put(x) for x in range(n_mem)]  # Fill queue with available memory indices
        self.current_jobs_in_process_list = [None for x in range(n_mem)]

        self.pool = multiprocessing.Pool(self.num_processes, initializer=init_worker,
                                         initargs=(self.sh_mem_rawarray, self.sh_mem_roi_rawarray,
                                                   (n_mem, self.image_size[0], self.image_size[1])))
        self.logger.info("Pool creation complete")

    def finish_processing(self):
        self.logger.info("{0}: Finish processing".format(self))
        self.finish_process_event.set()

    def stop_processes(self, terminate=True):
        self.logger.info("{0}: Stopping processes, terminate {1}".format(self, terminate))
        self.result_done_event.set()
        try:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
        except AttributeError:
            pass
        except AssertionError:
            pass

        self.logger.info("{0}: Processes stopped".format(self))

    def restart_processes(self):
        self.logger.info("{0}: Watchdog timeout, restating processes".format(self))
        while True:
            try:
                self.job_queue.get_nowait()
            except Queue.Empty:
                break
        self.processing_done()
        self.create_pool()


class TangoScanTask(Task):
    def __init__(self, scan_param, device_handler, name=None, timeout=None, trigger_dict=dict(), callback_list=list(),
                 read_callback=None):
        # type: (ScanParam) -> None
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.scan_param = scan_param
        self.device_handler = device_handler
        self.scan_result = None
        self.last_step_result = None
        self.read_callback = read_callback
        self.logger.debug("{0}: Scan parameters: {1}".format(self, scan_param))
        self.logger.debug("{0}: read_callback {1}".format(self, read_callback))

    def action(self):
        self.logger.info("{0} starting scan of {1} from {2} to {3}. ".format(self, self.scan_param.scan_attr_name,
                                                                             self.scan_param.scan_start_pos,
                                                                             self.scan_param.scan_end_pos))
        self.logger.info("{0} measuring {1}. ".format(self, self.scan_param.measure_attr_name_list))
        pos_list = list()
        pos_ind = 0
        timestamp_list = list()
        meas_list = list()
        next_pos = self.scan_param.scan_start_pos

        failed_steps = 0

        # Loop through the scan
        while self.get_done_event().is_set() is False:

            # Prepare set of tasks for this scan position.
            # - Write new pos
            # - Wait until arrived at new pos (monitor attribute)
            # - Read a set of measurements
            write_pos_task = TangoWriteAttributeTask(self.scan_param.scan_attr_name,
                                                     self.scan_param.scan_device_name,
                                                     self.device_handler,
                                                     next_pos,
                                                     name="write_pos",
                                                     timeout=self.timeout)
            monitor_pos_task = TangoMonitorAttributeTask(self.scan_param.scan_attr_name,
                                                         self.scan_param.scan_device_name,
                                                         self.device_handler,
                                                         next_pos,
                                                         tolerance=self.scan_param.scan_pos_tol,
                                                         interval=self.scan_param.scan_pos_check_interval,
                                                         name="monitor_pos",
                                                         timeout=self.timeout)
            measure_task_list = list()
            for meas_ind, meas_attr in enumerate(self.scan_param.measure_attr_name_list):
                # m_name = "read_{0}_{1}_{2}".format(meas_attr, pos_ind, self.last_step_result)
                m_name = "read_{0}_{1}_{2:.3f}_{3}".format(meas_attr, pos_ind, next_pos, meas_ind)
                self.logger.info("Measuring {0} on {1} using device handler {2}".format(meas_attr,
                                                                                        self.scan_param.measure_device_list[meas_ind],
                                                                                        self.device_handler))
                if self.read_callback is None:
                    read_task = TangoReadAttributeTask(meas_attr, self.scan_param.measure_device_list[meas_ind],
                                                       self.device_handler, name=m_name,
                                                       timeout=self.timeout)
                else:
                    self.logger.info("{0}: Reading {1}".format(self, m_name))
                    read_task = TangoReadAttributeTask(meas_attr, self.scan_param.measure_device_list[meas_ind],
                                                       self.device_handler, name=m_name,
                                                       timeout=self.timeout, callback_list=[self.read_callback])
                rep_task = RepeatTask(read_task, self.scan_param.measure_number, self.scan_param.measure_interval,
                                      name="rep_{0}".format(meas_attr), timeout=self.timeout)
                measure_task_list.append(rep_task)
            measure_bag_task = BagOfTasksTask(measure_task_list, name="measure_bag", timeout=self.timeout)
            step_sequence_task = SequenceTask([write_pos_task, monitor_pos_task, measure_bag_task], name="step_seq")
            step_sequence_task.start()
            step_result = step_sequence_task.get_result(wait=True, timeout=self.timeout)
            if step_sequence_task.is_cancelled() is True:
                failed_steps += 1
                if failed_steps > 3:
                    self.cancel()
                    return
            if self.is_cancelled() is True:
                return
            pos_list.append(step_result[1].value)
            timestamp_list.append(step_result[1].time)
            meas_list.append(step_result[2])
            self.last_step_result = step_result
            self.result = step_result
            # Step done, notify callbacks:
            if self.is_done() is False:
                self.logger.debug("{0} Calling {1} callbacks".format(self, len(self.callback_list)))
                for callback in self.callback_list:
                    callback(self)
            next_pos += self.scan_param.scan_step
            pos_ind += 1
            if next_pos > self.scan_param.scan_end_pos:
                self.event_done.set()

        self.scan_result = ScanResult(pos_list, meas_list, timestamp_list)
        self.result = self.scan_result

    def get_last_step_result(self):
        return self.last_step_result


class PopulateDeviceListTask(Task):
    """
    Populate matching section data by quering tango database properties.

    This does not establish a connection with the device servers. That is done by
    the device handler.

    Data retrieved:
    Quads... name, length, position, polarity
    Screens... name, position
    """

    def __init__(self, sections, name=None, action_exec_type="thread", timeout=None, trigger_dict=dict(),
                 callback_list=list()):
        Task.__init__(self, name, action_exec_type="thread", timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.sections = sections

    def action(self):
        self.logger.info("{0} Populating matching sections by checking tango database.".format(self))
        db = pt.Database()
        self.logger.debug("{0} Database retrieved".format(self))

        sections = self.sections
        sect_quads = dict()
        sect_screens = dict()

        # Loop through sections to find matching devices based on their names:
        for s in sections:
            # Quad names are e.g. i-ms1/mag/qb-01
            quad_dev_list = db.get_device_exported("*{0}*/mag/q*".format(s)).value_string
            quad_list = list()
            for mag_name in quad_dev_list:
                quad = dict()
                try:
                    # Extract data for each found quad:

                    name = mag_name.split("/")[-1].lower()
                    p = db.get_device_property(mag_name, ["__si", "length", "polarity", "circuitproxies"])
                    position = np.double(p["__si"][0])
                    length = np.double(p["length"][0])
                    polarity = np.double(p["polarity"][0])
                    crq = p["circuitproxies"][0]
                    quad = SectionQuad(name, position, length, mag_name, crq, polarity)
                    quad_list.append(quad)
                except IndexError as e:
                    self.logger.exception("Index error when parsing quad {0}: ".format(mag_name))
                    # self.logger.error("Index error when parsing quad {0}: {1}".format(q, e))
                    pass
                except KeyError as e:
                    self.logger.error("Key error when parsing quad {0}: {1}".format(mag_name, e))
                    pass

            # Screen names are e.g. i-ms1/dia/scrn-01
            screen_dev_list = db.get_device_exported("*{0}*/dia/scrn*".format(s)).value_string
            screen_list = list()
            for sc_name in screen_dev_list:
                scr = dict()
                try:
                    # Extract data for each found screen
                    name = sc_name.split("/")[-1].lower()
                    lima_name = sc_name.replace("/", "-")
                    position = np.double(db.get_device_property(sc_name, "__si")["__si"][0])
                    liveviewer = "lima/liveviewer/{0}".format(lima_name)
                    beamviewer = "lima/beamviewer/{0}".format(lima_name)
                    limaccd = "lima/limaccd/{0}".format(lima_name)
                    scr = SectionScreen(name, position, liveviewer, beamviewer, limaccd, sc_name)
                    screen_list.append(scr)
                # If name and/or position for the screen is not retrievable we cannot use it:
                except IndexError as e:
                    self.logger.exception("Index error when parsing screen {0}. "
                                          "This screen will not be included in the populated list. ".format(mag_name))
                    pass
                except KeyError as e:
                    self.logger.exception("Key error when parsing screen {0}"
                                          "This screen will not be included in the populated list. ".format(mag_name))
                    pass

            sect_quads[s] = quad_list
            sect_screens[s] = screen_list
            self.logger.debug("{0} Populating section {1}:\n "
                              "    Found quads: {2} \n "
                              "    Found screens: {3} \n"
                              "--------------------------------------------------------\n"
                              "".format(self, s.upper(), quad_list, screen_list))
        self.result = SectionDevices(sect_quads, sect_screens)


class PopulateDummyDeviceList(Task):
    """
    Populate matching section data by assuming dummy devices with no database.

    This does not establish a connection with the device servers. That is done by
    the device handler. The data must be retrieved through device attributes since properties
    don't work without a database.

    Data retrieved:
    Quads... name, length, position, polarity
    Screens... name, position
    """

    def __init__(self, sections, dummy_name_dict, device_handler, name=None, action_exec_type="thread",
                 timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, action_exec_type="thread", timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.sections = sections
        self.dummy_name_dict = dummy_name_dict
        self.device_handler = device_handler

    def action(self):
        self.logger.info("{0} Populating matching sections by assuming dummy devices.".format(self))

        sections = self.sections
        sect_quads = dict()
        sect_screens = dict()

        # Loop through sections to find matching devices based on their names:
        for s in sections:
            # Quad names are e.g. i-ms1/mag/qb-01
            sect_dict = self.dummy_name_dict[s]
            quad_dev_list = sect_dict["mag"]
            quad_list = list()
            for mag_name in quad_dev_list:
                quad = dict()
                try:
                    # Extract data for each found quad:
                    quad_dev = self.device_handler.get_device(mag_name)
                    name = mag_name.split("/")[-1].lower()
                    position = quad_dev.position
                    length = quad_dev.ql
                    polarity = 1.0
                    #crq = self.dummy_name_dict["crq"]
                    crq = mag_name
                    quad = SectionQuad(name, position, length, mag_name, crq, polarity)
                    quad_list.append(quad)
                except IndexError as e:
                    self.logger.exception("Index error when parsing quad {0}: ".format(mag_name))
                    # self.logger.error("Index error when parsing quad {0}: {1}".format(q, e))
                    pass
                except KeyError as e:
                    self.logger.error("Key error when parsing quad {0}: {1}".format(mag_name, e))
                    pass

            # Screen names are e.g. i-ms1/dia/scrn-01
            screen_dev_list = [sect_dict["screen"]]
            screen_list = list()
            for sc_name in screen_dev_list:
                scr = dict()
                try:
                    # Extract data for each found screen
                    name = sc_name.split("/")[-1].lower()
                    position = 10.0
                    liveviewer = sect_dict["liveviewer"]
                    beamviewer = sect_dict["beamviewer"]
                    limaccd = sect_dict["limaccd"]
                    scr = SectionScreen(name, position, liveviewer, beamviewer, limaccd, sc_name)
                    screen_list.append(scr)
                # If name and/or position for the screen is not retrievable we cannot use it:
                except IndexError as e:
                    self.logger.exception("Index error when parsing screen {0}: ".format(mag_name))
                    pass
                except KeyError as e:
                    self.logger.exception("Key error when parsing screen {0}: ".format(mag_name))
                    pass

            sect_quads[s] = quad_list
            sect_screens[s] = screen_list
            self.logger.debug("{0} Populating section {1}:\n "
                              "    Found quads: {2} \n "
                              "    Found screens: {3} \n"
                              "--------------------------------------------------------\n"
                              "".format(self, s.upper(), quad_list, screen_list))
        self.result = SectionDevices(sect_quads, sect_screens)


class FitQuadDataTask(Task):
    """
    Fit supplied quad data and calculate beam twiss parameters
    """

    def __init__(self, processed_image_list, accelerator_params, algo="full", axis="x",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.processed_image_list = processed_image_list    # type: list of ProcessedImage
        # K value for each image is stored in the image list
        self.accelerator_params = accelerator_params        # type: AcceleratorParameters
        self.algo = algo
        self.axis = axis
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action.".format(self))
        t0 = time.time()
        if self.algo == "full":
            fitresult = self.fit_full_transfer_matrix()
        else:
            fitresult = self.fit_thin_lens()
        self.result = fitresult
        self.logger.debug("{0}: Fit time {1:.2f} s".format(self, time.time()-t0))

    def fit_thin_lens(self):
        self.logger.info("Fitting image data using thin lens approximation")
        k_data = np.array([pi.k_value for pi in self.processed_image_list]).flatten()
        if self.axis == "x":
            sigma_data = np.array([pi.sigma_x for pi in self.processed_image_list]).flatten()
        else:
            sigma_data = np.array([pi.sigma_y for pi in self.processed_image_list]).flatten()
        en_data = np.array([pi.enabled for pi in self.processed_image_list]).flatten()
        try:
            s2 = (sigma_data[en_data]) ** 2
        except IndexError as e:
            self.logger.warning("Could not address enabled sigma values. "
                                "En_data: {0}, sigma_data: {1}".format(en_data, sigma_data))
            self.result = e
            self.cancel()
            return e

        k = k_data[en_data]
        ind = np.isfinite(s2)
        p = np.polyfit(k[ind], s2[ind], 2, full=True)
        poly = p[0]
        res = p[1]
        self.logger.debug("Fit coefficients: {0}".format(poly))
        d = self.accelerator_params.quad_screen_dist
        L = self.accelerator_params.quad_length
        gamma = self.accelerator_params.electron_energy / 0.511
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

        x = np.linspace(k.min(), k.max(), 100)
        y = np.polyval(poly, x)
        fit_data = [x, np.sqrt(y)]
        fitresult = FitResult(poly=poly, alpha=alpha, beta=beta, eps_n=eps_n, eps=eps, gamma_e=gamma,
                              fit_data=fit_data, residual=res)
        return fitresult

    def fit_full_transfer_matrix(self):
        self.logger.info("Fitting using full transfer matrix")
        k_data = np.array([pi.k_value for pi in self.processed_image_list]).flatten()
        if self.axis == "x":
            sigma_data = np.array([pi.sigma_x for pi in self.processed_image_list]).flatten()
        else:
            sigma_data = np.array([pi.sigma_y for pi in self.processed_image_list]).flatten()
        en_data = np.array([pi.enabled for pi in self.processed_image_list]).flatten()
        d = self.accelerator_params.quad_screen_dist
        L = self.accelerator_params.quad_length
        gamma_energy = self.accelerator_params.electron_energy / 0.511
        self.logger.debug("sigma_data: {0}".format(sigma_data))
        self.logger.debug("en_data: {0}".format(en_data))
        self.logger.debug("k_data: {0}".format(k_data))
        try:
            s2 = (sigma_data[en_data]) ** 2
        except IndexError as e:
            self.logger.warning("Could not address enabled sigma values. "
                                "En_data: {0}, sigma_data: {1}".format(en_data, sigma_data))
            self.result = e
            self.cancel()
            return e

        k = k_data[en_data]
        ind = np.isfinite(s2)

        k_sqrt = np.sqrt(k[ind]*(1+0j))
        # self.logger.debug("k_sqrt = {0}".format(k_sqrt))
        # Matrix elements for single quad + drift:
        A = np.real(np.cos(k_sqrt * L) - d * k_sqrt * np.sin(k_sqrt * L))
        B = np.real(1 / k_sqrt * np.sin(k_sqrt * L) + d * np.cos(k_sqrt * L))
        M = np.vstack((A*A, -2*A*B, B*B)).transpose()
        try:
            l_data = np.linalg.lstsq(M, s2[ind], rcond=-1)
            x = l_data[0]
            res = l_data[1]
        except Exception as e:
            self.logger.error("Error when fitting lstsqr: {0}".format(e))
            self.result = e
            self.cancel()
            return e
        self.logger.debug("Fit coefficients: {0}".format(x[0]))
        eps = np.sqrt(x[2] * x[0] - x[1]**2)
        eps_n = eps * gamma_energy
        beta = x[0] / eps
        alpha = x[1] / eps

        self.logger.info("-------------------------------")
        self.logger.info("eps_n  = {0:.3f} mm x mrad".format(eps_n * 1e6))
        self.logger.info("beta   = {0:.4g} m".format(beta))
        self.logger.info("alpha  = {0:.4g} rad".format(alpha))
        self.logger.info("-------------------------------")

        try:
            x = np.linspace(k.min(), k.max(), 100)
        except ValueError as e:
            self.logger.exception("{0} k_data error:".format(self))
            self.result = e
            self.cancel()
            return e
        x_sqrt = np.sqrt(x*(1+0j))
        Ax = np.real(np.cos(x_sqrt * L) - d * x_sqrt * np.sin(x_sqrt * L))
        Bx = np.real(1 / x_sqrt * np.sin(x_sqrt * L) + d * np.cos(x_sqrt * L))

        y = Ax**2 * beta * eps - 2 * Ax * Bx * alpha * eps + Bx**2 * (1 + alpha**2) * eps / beta
        fit_data = [x, np.sqrt(y)]
        fitresult = FitResult(poly=None, alpha=alpha, beta=beta, eps_n=eps_n, eps=eps, gamma_e=gamma_energy,
                              fit_data=fit_data, residual=res)
        return fitresult


class DeviceHandler(object):
    """
    Handler for open devices.
    Devices are stored in a dict for easy retrieval.
    New devices are added asynchronously with add_device method.
    """
    def __init__(self, tango_host=None, name=None):
        self.devices = dict()
        self.tango_host = tango_host
        self.timeout = 10.0
        if name is None:
            self.name = self.__repr__()
        else:
            self.name = name
        self.logger = logging.getLogger("Task.{0}".format(self.name.upper()))
        self.logger.setLevel(logging.INFO)

    def get_device(self, device_name):
        self.logger.debug("{0} Returning device {1}".format(self, device_name))
        try:
            dev = self.devices[device_name]
        except KeyError:
            # Maybe this should just raise an exception instead of auto-adding:
            task = self.add_device(device_name)
            dev = task.get_result(wait=True, timeout=self.timeout)
            if task.is_cancelled():
                raise pt.DevFailed(dev)
        return dev

    def add_device(self, device_name):
        """
        Add a device to the open devices dictionary.
        A device connect task is created and started.

        :param device_name: Tango name of device
        :return: opened device proxy
        """
        self.logger.info("{0} Adding device {1} to device handler".format(self, device_name))
        if device_name in self.devices:
            self.logger.debug("Device already in dict. No need")
            task = Task(name="DEV_DONE")
            task.start()
            task.get_result(wait=True)
            task.result = self.devices[device_name]
            return task
        if self.tango_host is not None:
            full_dev_name = "{0}/{1}".format(self.tango_host, device_name)
        else:
            full_dev_name = device_name

        # Create task that connects to device, then trigger another task that adds it to device dict:
        task = TangoDeviceConnectTask(full_dev_name, name="CONNECT_{0}".format(device_name))
        task.start()

        task_call = CallableTask(self._dev_connect_done, (device_name, task),
                                 name="ADD_{0}".format(device_name))
        task_call.add_trigger(task)
        task_call.start()

        return task_call

    def add_devices(self, device_names):
        agg_task = DelayTask(0.0)
        for dn in device_names:
            t = self.add_device(dn)
            agg_task.add_trigger(t)
        agg_task.start()
        return agg_task

    def _dev_connect_done(self, device_name, task):
        dev = task.get_result(wait=True, timeout=self.timeout)
        self.logger.info("{0} {1} Device connection completed. Returned {1}".format(self, device_name, dev))
        self.devices[device_name] = dev
        return dev

    def __str__(self):
        s = "{0} {1}".format(type(self).__name__, self.name)
        return s


def test_f(in_data):
    t = time.time()
    time.sleep(0.5)
    s = "{0}: {1}".format(t, in_data)
    logger.info(s)
    return s


if __name__ == "__main__":
    tests = ["delay", "dev_handler", "exc", "monitor", "load_im", "load_im_dir", "proc_im",
             "scan", "fit", "populate", "populate_dummy"]
    test = "populate_dummy"
    if test == "delay":
        t1 = DelayTask(2.0, name="task1")
        t2 = DelayTask(1.0, name="task2", trigger_dict={"delay": t1})
        t3 = DelayTask(0.5, name="task3", trigger_dict={"delay1": t2, "delay2": t1})
        t4 = DelayTask(0.1, name="task4")
        t3.add_trigger(t4)
        t2.timeout = 1.0
        t4.start()
        time.sleep(0.5)
        t3.start()
        t2.start()
        t1.start()
    elif test == "dev_handler":
        handler = DeviceHandler("b-v0-gunlaser-csdb-0:10000", name="Handler")
        dev_name = "sys/tg_test/1"
        t1 = handler.add_device(dev_name)
        t2 = TangoReadAttributeTask("double_scalar", dev_name, handler,
                                    name="read_ds", trigger_dict={"1": t1})
        t2.start()

        t3 = TangoWriteAttributeTask("double_scalar_w", dev_name, handler, 10.0, trigger_dict={"1": t2})
        t4 = TangoReadAttributeTask("double_scalar_w", dev_name, handler, trigger_dict={"1": t3})
        t4.start()
        t3.start()
        logger.info("Double scalar: {0}".format(t4.get_result(wait=False)))
        logger.info("Double scalar: {0}".format(t4.get_result(True).value))

        t6 = SequenceTask([t2, DelayTask(1.0, "delay_seq")], name="seq")
        t5 = RepeatTask(t6, 5, name="rep")
        t5.start()
        time.sleep(2)
        t6.cancel()

    elif test == "exc":
        t1 = DelayTask(2.0, name="long delay")
        t1.start()
        time.sleep(1.0)
        t1.cancel()

    elif test == "monitor":
        handler = DeviceHandler("b-v0-gunlaser-csdb-0:10000", name="Handler")
        dev_name = "sys/tg_test/1"
        th = handler.add_device(dev_name)
        th.get_result(wait=False)
        t1 = TangoMonitorAttributeTask("double_scalar_w", "sys/tg_test/1", handler, target_value=100, tolerance=0.01,
                                       tolerance_type="rel", interval=0.5, name="monitor")
        t1.start()

    elif test == "load_im":
        image_name = "03_03_1.035_.png"
        path_name = "D:\\Programmering\emittancescansinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t1 = LoadQuadImageTask(image_name, path_name, name="load_im")
        t1.start()

    elif test == "load_im_dir":
        image_name = "03_03_1.035_.png"
        path_name = "..\\..\\emittancesinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t1 = LoadQuadScanDirTask(path_name, "quad_dir", process_exec_type="thread")
        t1.start()

    elif test == "proc_im":

        t1 = ProcessPoolTask(time.sleep)
        t1.start()

    elif test == "scan":
        handler = DeviceHandler("b-v0-gunlaser-csdb-0:10000", name="Handler")
        dev_name = "sys/tg_test/1"
        scan_param = ScanParam("double_scalar_w", dev_name, scan_start_pos=0.0, scan_end_pos=5.0, scan_step=1.0,
                               scan_pos_check_interval=0.1, scan_pos_tol=0.01,
                               measure_attr_name_list=["double_scalar", "double_scalar_w"],
                               measure_device_list=[dev_name, dev_name], measure_number=3, measure_interval=0.5)
        t1 = TangoScanTask(scan_param, handler, name="scan_task", timeout=3.0)
        t1.start()

    elif test == "fit":
        path_name = "..\\..\\emittancesinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        # path_name = "D:\\Programmering\emittancescansinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t2 = LoadQuadScanDirTask(path_name, process_exec_type="thread", kernel_size=3, name="quad_dir")
        t2.start()
        quad_scan_data = t2.get_result(True)    # type: QuadScanData
        acc_params = quad_scan_data.acc_params
        quad_images = quad_scan_data.proc_images
        t3 = FitQuadDataTask(quad_images, acc_params, "full", name="fit_data")
        t3.start()
        t4 = ProcessAllImagesTask(quad_scan_data, process_exec_type="thread", name="proc_all")
        t4.start()

    elif test == "populate":
        t1 = PopulateDeviceListTask(["MS1", "MS2"], name="pop", action_exec_type="process")
        t1.start()
        logger.info("Task started, doing something else")
        time.sleep(0.5)
        logger.info("Ok, then")
        # logger.info("Populate returned: {0}".format(t1.get_result(True)))

    elif test == "populate_dummy":
        section_list = ["MS1", "MS2"]
        device_handler = DeviceHandler(name="Handler")
        ms1_dict = {"mag": ["127.0.0.1:10000/i-ms1/mag/qb-01#dbase=no",
                            "127.0.0.1:10000/i-ms1/mag/qb-02#dbase=no",
                            "127.0.0.1:10000/i-ms1/mag/qb-03#dbase=no",
                            "127.0.0.1:10000/i-ms1/mag/qb-04#dbase=no"],
                    "crq": "127.0.0.1:10000/i-ms1/mag/qb-01#dbase=no",
                    "screen": "127.0.0.1:10001/i-ms1/dia/scrn-01#dbase=no",
                    "beamviewer": "127.0.0.1:10003/lima/beamviewer/i-ms1-dia-scrn-01#dbase=no",
                    "liveviewer": "127.0.0.1:10002/lima/liveviewer/i-ms1-dia-scrn-01#dbase=no",
                    "limaccd": "127.0.0.1:10004/lima/limaccd/i-ms1-dia-scrn-01#dbase=no"}

        ms2_dict = {"mag": ["127.0.0.1:10000/i-ms2/mag/qb-01#dbase=no",
                            "127.0.0.1:10000/i-ms2/mag/qb-02#dbase=no",
                            "127.0.0.1:10000/i-ms2/mag/qb-03#dbase=no",
                            "127.0.0.1:10000/i-ms2/mag/qb-04#dbase=no"],
                    "crq": "127.0.0.1:10000/i-ms2/mag/qb-01#dbase=no",
                    "screen": "127.0.0.1:10001/i-ms2/dia/scrn-02#dbase=no",
                    "beamviewer": "127.0.0.1:10003/lima/beamviewer/i-ms2-dia-scrn-02#dbase=no",
                    "liveviewer": "127.0.0.1:10002/lima/liveviewer/i-ms2-dia-scrn-02#dbase=no",
                    "limaccd": "127.0.0.1:10004/lima/limaccd/i-ms2-dia-scrn-02#dbase=no"}

        ms3_dict = {"mag": ["127.0.0.1:10000/i-ms3/mag/qf-01#dbase=no",
                            "127.0.0.1:10000/i-ms3/mag/qf-02#dbase=no",
                            "127.0.0.1:10000/i-ms3/mag/qf-03#dbase=no",
                            "127.0.0.1:10000/i-ms3/mag/qf-04#dbase=no"],
                    "crq": "127.0.0.1:10000/i-ms3/mag/qb-01#dbase=no",
                    "screen": "127.0.0.1:10001/i-ms3/dia/scrn-01#dbase=no",
                    "beamviewer": "127.0.0.1:10003/lima/beamviewer/i-ms3-dia-scrn-01#dbase=no",
                    "liveviewer": "127.0.0.1:10002/lima/liveviewer/i-ms3-dia-scrn-01#dbase=no",
                    "limaccd": "127.0.0.1:10004/lima/limaccd/i-ms3-dia-scrn-01#dbase=no"}

        dummy_name_dict = {"MS1": ms1_dict, "MS2": ms2_dict, "MS3": ms3_dict, "SP02": ms3_dict}

        t1 = PopulateDummyDeviceList(sections=section_list, dummy_name_dict=dummy_name_dict,
                                     device_handler=device_handler, name="pop_sections", action_exec_type="process")
        t1.start()
        logger.info("Task started, doing something else")
        time.sleep(0.5)
        logger.info("Ok, then")
        # logger.info("Populate returned: {0}".format(t1.get_result(True)))
