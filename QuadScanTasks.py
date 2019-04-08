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
except ImportError:
    try:
        import tango as pt
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
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler

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
        try:
            attr = dev.read_attribute(self.attribute_name)
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
            self.cancel()
            return
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

        self.result = read_task.get_result(wait=False)


class LoadQuadImageTask(Task):
    def __init__(self, image_name, path=".", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.image_name = image_name
        self.path = path
        self.logger.setLevel(logging.DEBUG)

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
        image = np.array(PIL.Image.open(filename))
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
    def __init__(self, quadscandir, process_now=True, threshold=None, kernel_size=3,
                 image_processor_task=None, process_exec_type="process",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.DEBUG)

        self.pathname = quadscandir
        self.process_now = process_now
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.processed_image_list = list()
        self.task_seq = None
        # The images are processed as they are loaded:
        self.image_processor = None         # type: ImageProcessorTask
        if image_processor_task is None:
            self.image_processor = ImageProcessorTask(threshold=threshold, kernel=kernel_size,
                                                      process_exec=process_exec_type,
                                                      trigger_dict=trigger_dict, name="loaddir_image_proc")
        else:
            # If the image_processor was supplied, don't add self as trigger.
            self.image_processor = image_processor_task
        if self.image_processor.is_started() is False:
            self.logger.info("Starting image_processor")
            self.image_processor.start()
        self.image_processor.add_callback(self.processed_image_done)    # Call this method after completing each image

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
        data_dict["roi_center"] = [np.double(rc[1]), np.double(rc[0])]
        rd = data_dict["roi_dim"].split(" ")
        data_dict["roi_dim"] = [np.double(rd[1]), np.double(rd[0])]
        try:
            data_dict["bpp"] = np.int(data_dict["bpp"])
        except KeyError:
            data_dict["bpp"] = 16
            self.logger.debug("{1} Loaded data_dict: \n{0}".format(pprint.pformat(data_dict), self))

        self.image_processor.set_roi(data_dict["roi_center"], data_dict["roi_dim"])
        self.image_processor.set_processing_parameters(self.threshold, data_dict["pixel_size"], self.kernel_size)
        file_list = os.listdir(load_dir)
        image_file_list = list()
        load_task_list = list()         # List of tasks, each loading an image. Loading should be done in sequence
                                        # as this in not sped up by paralellization
        for file_name in file_list:
            if file_name.endswith(".png"):
                image_file_list.append(file_name)
                t = LoadQuadImageTask(file_name, load_dir, name=file_name,
                                      callback_list=[self.image_processor.process_image])
                t.logger.setLevel(logging.WARNING)
                load_task_list.append(t)

        self.logger.debug("{1} Found {0} images in directory".format(len(image_file_list), self))
        self.task_seq = SequenceTask(load_task_list, name="load_seq")
        self.task_seq.start()
        # Wait for image sequence to be done reading:
        image_list = self.task_seq.get_result(wait=True)
        if self.task_seq.is_cancelled():
            self.logger.error("Load image error: {0}".format(image_list))
        # Now wait for images to be done processing:
        # The processed images are already stored in the processed_image_list by the callback
        self.logger.debug("{0}: Waiting for image processing to finish".format(self))
        starttime = time.time()
        dt = 0
        while self.image_processor.pending_images_in_queue > 0 and dt < self.timeout:
            dt = time.time() - starttime
            time.sleep(0.01)
        # self.image_processor.wait_for_queue_empty()
        self.logger.debug("{0}: Image processing finished".format(self))

        acc_params = AcceleratorParameters(electron_energy=float(data_dict["beam_energy"]),
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

        self.result = QuadScanData(acc_params, image_list, self.processed_image_list)
        self.image_processor.clear_callback_list()

    def processed_image_done(self, image_processor_task):
        # type: (ImageProcessorTask) -> None
        proc_image = image_processor_task.get_result(wait=False)    # type: ProcessedImage
        if image_processor_task.is_cancelled():
            self.result = proc_image
            self.cancel()
            return
        if image_processor_task.is_done() is False:
            if isinstance(proc_image, Exception):
                self.logger.error("{0}: Found error in processed image: {1}".format(self, proc_image))
            else:
                self.logger.debug("Adding processed image {0} {1} to list".format(proc_image.k_ind, proc_image.image_ind))
                self.processed_image_list.append(proc_image)

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
        self.logger.setLevel(logging.DEBUG)

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
        self.logger.setLevel(logging.DEBUG)

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


class ImageProcessorTask2(Task):
    def __init__(self, roi_cent=None, roi_dim=None, threshold=None, cal=[1.0, 1.0], kernel=3, process_exec="process",
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self,  name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.DEBUG)

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
        self.logger.setLevel(logging.DEBUG)

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
            self.logger.info("Starting image_processor")
            self.image_processor.start()
        self.image_processor.add_callback(self.processed_image_done)
        self.processed_image_list = list()

    def action(self):
        self.logger.info("{0}: entering action".format(self))
        self.processed_image_list = list()

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
            self.logger.debug("Adding processed image {0} {1} to list".format(proc_image.k_ind, proc_image.image_ind))
            with self.lock:
                self.processed_image_list.append(proc_image)
                self.pending_images -= 1
                if self.pending_images <= 0:
                    self.images_done_event.set()
        else:
            self.images_done_event.set()


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
        while self.get_done_event().is_set() is False:
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
                m_name = "read_{0}_{1}_{2}_{3}".format(meas_attr, pos_ind, next_pos, meas_ind)
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
    the device handler.

    Data retrieved:
    Quads... name, length, position, polarity
    Screens... name, position
    """

    def __init__(self, sections, dummy_name_dict, name=None, action_exec_type="thread",
                 timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, action_exec_type="thread", timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.sections = sections
        self.dummy_name_dict = dummy_name_dict

    def action(self):
        self.logger.info("{0} Populating matching sections by assuming dummy devices.".format(self))

        sections = self.sections
        sect_quads = dict()
        sect_screens = dict()

        # Loop through sections to find matching devices based on their names:
        for s in sections:
            # Quad names are e.g. i-ms1/mag/qb-01
            quad_dev_list = [self.dummy_name_dict["mag"]]
            quad_list = list()
            for mag_name in quad_dev_list:
                quad = dict()
                try:
                    # Extract data for each found quad:

                    name = mag_name.split("/")[-1].lower()
                    position = 5.0
                    length = 0.2
                    polarity = 1.0
                    crq = self.dummy_name_dict["crq"]
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
            screen_dev_list = [self.dummy_name_dict["screen"]]
            screen_list = list()
            for sc_name in screen_dev_list:
                scr = dict()
                try:
                    # Extract data for each found screen
                    name = sc_name.split("/")[-1].lower()
                    position = 10.0
                    liveviewer = self.dummy_name_dict["liveviewer"]
                    beamviewer = self.dummy_name_dict["beamviewer"]
                    limaccd = self.dummy_name_dict["limaccd"]
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
        except IndexError:
            self.logger.warning("Could not address enabled sigma values. "
                                "En_data: {0}, sigma_data: {1}".format(en_data, sigma_data))
            return
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
        # self.logger.debug("k_sqrt = {0}".format(k_sqrt))
        A = np.real(np.cos(k_sqrt * L) - d * k_sqrt * np.sin(k_sqrt * L))
        B = np.real(1 / k_sqrt * np.sin(k_sqrt * L) + d * np.cos(k_sqrt * L))
        M = np.vstack((A*A, -2*A*B, B*B)).transpose()
        try:
            l_data = np.linalg.lstsq(M, s2[ind], rcond=-1)
            x = l_data[0]
            res = l_data[1]
        except Exception as e:
            self.logger.error("Error when fitting lstsqr: {0}".format(e))
            return
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

        x = np.linspace(k.min(), k.max(), 100)
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
            return True
        if self.tango_host is not None:
            full_dev_name = "{0}/{1}".format(self.tango_host, device_name)
        else:
            full_dev_name = device_name
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
             "scan", "fit", "populate"]
    test = "load_im_dir"
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
