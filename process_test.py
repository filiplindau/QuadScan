import multiprocessing
try:
    import Queue
except ModuleNotFoundError:
    import queue as Queue

import numpy as np
from tasks.GenericTasks import Task
import threading
import time
from scipy.signal import medfilt2d
from QuadScanDataStructs import *

import logging
logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def clearable_pool_worker(in_queue, out_queue):
    while True:
        task = in_queue.get()
        f = task[0]
        args = task[1]
        kwargs = task[2]
        proc_id = task[3]
        logger.debug("Pool worker executing {0} with args {1}, {2}".format(f, args, kwargs))
        logger.debug("Pool worker executing {0} ".format(f))
        try:
            retval = f(*args, **kwargs)
            logger.debug("Putting {0} on out_queue".format(retval))
            # logger.debug("Putting {0} result on out_queue".format(f))
        except Exception as e:
            retval = e
            logger.error("{0} Error {1} ".format(f, retval))
        out_queue.put((retval, proc_id))


class ProcessPoolTask(Task):
    """
    Start up a process pool that can consume data. Emit trigger when queue is empty.

    The results of the tasks are stored in a list.
    """

    def __init__(self, work_func, number_processes=multiprocessing.cpu_count(), name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.work_func = work_func
        self.lock = multiprocessing.Lock()
        self.finish_process_event = threading.Event()
        self.pending_work_items = list()

        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.num_processes = number_processes
        self.processes = None
        self.next_process_id = 0
        self.completed_work_items = 0
        self.id_lock = threading.Lock()

        self.result_thread = None
        self.stop_result_thread_flag = False
        self.result_dict = dict()

        self.logger.setLevel(logging.DEBUG)

    def run(self):
        self.create_processes()
        Task.run(self)

    def start(self):
        self.next_process_id = 0
        Task.start(self)

    def action(self):
        self.logger.info("{0}: starting processing pool for "
                         "executing function {1} across {2} processes.".format(self, self.work_func, self.num_processes))
        self.logger.info("{1}: Starting {0} processes".format(self.num_processes, self))
        self.completed_work_items = 0
        for p in self.processes:
            p.start()
        self.stop_result_thread_flag = False
        self.result_thread = threading.Thread(target=self.result_thread_func)
        self.result_thread.start()

        self.finish_process_event.wait(self.timeout)
        if self.finish_process_event.is_set() is False:
            self.cancel()
            return
        t0 = time.time()
        while self.completed_work_items < self.next_process_id:
            # self.logger.debug("{0}: Waiting for {1} work items".format(self, self.next_process_id - self.completed_work_items))
            time.sleep(0.01)
            if time.time() - t0 > 5.0:
                self.logger.error("{0}: Timeout waiting for {1} work items to complete".format(self, self.next_process_id - self.completed_work_items))
                self.stop_processes(terminate=True)
                self.result = self.result_dict
                return
        self.stop_processes(terminate=False)
        self.result = self.result_dict

    def add_work_item(self, *args, **kwargs):
        self.logger.debug("{0}: Adding work item".format(self))
        # self.logger.debug("{0}: Args: {1}, kwArgs: {2}".format(self, args, kwargs))
        proc_id = self.next_process_id
        self.next_process_id += 1
        self.logger.debug("{0}: queue size: {1}".format(self, self.in_queue.qsize()))
        self.in_queue.put((self.work_func, args, kwargs, proc_id))
        self.result_dict[proc_id] = None
        # self.logger.debug("{0}: Work item added to queue. Process id: {1}".format(self, proc_id))

    def create_processes(self):
        if self.processes is not None:
            self.stop_processes()

        self.logger.info("{1}: Creating {0} processes".format(self.num_processes, self))
        p_list = list()
        for p in range(self.num_processes):
            p = multiprocessing.Process(target=clearable_pool_worker, args=(self.in_queue, self.out_queue))
            p_list.append(p)
        self.processes = p_list

    def stop_processes(self, terminate=True):
        self.logger.info("{0}: Stopping processes".format(self))
        if terminate is True:
            if self.processes is not None:
                for p in self.processes:
                    p.terminate()
        # self.processes = None
        self.stop_result_thread_flag = True
        try:
            self.result_thread.join(1.0)
        except AttributeError:
            pass
        self.result_thread = None
        self.logger.info("{0}: Processes stopped".format(self))

    def finish_processing(self):
        self.finish_process_event.set()

    def cancel(self):
        self.finish_processing()
        Task.cancel(self)

    def clear_pending_tasks(self):
        self.logger.info("{0}: Clearing pending tasks".format(self))
        while self.in_queue.empty() is False:
            try:
                self.logger.debug("get no wait")
                work_item = self.in_queue.get_nowait()
                proc_id = work_item[3]
                self.logger.debug("Removing task {0}".format(id))
                try:
                    self.result_dict.pop(proc_id)
                except KeyError:
                    pass
            except multiprocessing.queues.Empty:
                self.logger.debug("In-queue empty")
                break

    def result_thread_func(self):
        """
        This method runs in a thread collecting the results from the worker processes.
        :return:
        """
        self.logger.debug("{0}: Starting result collection thread".format(self))
        while self.stop_result_thread_flag is False:
            # Wait for a new result item:
            try:
                result = self.out_queue.get(True, 0.1)
            except multiprocessing.queues.Empty:
                continue
            # Get result and which process it was that sent the result:
            self.logger.debug("Got result from process. {0}".format(type(result)))
            retval = result[0]
            proc_id = result[1]
            self.completed_work_items += 1
            try:
                with self.lock:
                    self.result_dict[proc_id] = retval
            except KeyError as e:
                self.logger.error("Work item id {0} not found.".format(proc_id))
                raise e
            self.result = retval
            # The clearable_pool_worker will put any encountered exception as the result value in the output queue:
            if isinstance(retval, Exception):
                pass    # Pass through for now...
                # raise retval
            # self.logger.debug("{0}: result_dict {1}".format(self, pprint.pformat(self.result_dict)))
            for callback in self.callback_list:
                callback(self)
        self.logger.debug("{0}: Exiting result_thread_func".format(self))


def f(image, enabled=True):
    print("Enabled {0}".format(enabled))
    if enabled:
        s = image.sum()
    else:
        1/0
    return s


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
    try:
        x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
        y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])

        # Extract ROI and convert to double:
        pic_roi = np.double(image[x[0]:x[1], y[0]:y[1]])
    except IndexError:
        pic_roi = np.double(image)
    n = 2 ** bpp

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
    #
    # result = ProcessedImage(k_ind=k_ind, k_value=k_value, image_ind=image_ind, pic_roi=0,
    #                         line_x=0, line_y=0, x_cent=0, y_cent=0,
    #                         sigma_x=0, sigma_y=0, q=0, enabled=enabled, threshold=threshold)
    logger.debug("Image {0}, {1} processed in pool, time {2:.2f} ms".format(k_ind, image_ind, 1e3*(time.time()-t0)))
    return result


if __name__ == "__main__":

    t = ProcessPoolTask(f, name="test")
    t.start()

    im = np.random.random((1000, 1000))
    t.add_work_item(im, enabled=True)

    # t.add_work_item(im, enabled=False)

    time.sleep(2.0)

    res = t.get_result(wait=False)
    logger.info("RESULT from task: {0}, {1}".format(res, type(res)))

    t.stop_processes()
    t.cancel()
    time.sleep(1.0)

    # Process image func
    t = ProcessPoolTask(process_image_func, name="test")
    t.start()

#    im = np.random.random((1000, 1000))
    k_ind = 0
    k_val = 0
    im_ind = 0
    th = 0
    roi_c = [0, 0]
    roi_d = [100, 100]
    cal = [1, 1]
    kern = 3
    norm = False
    en = True
#    process_image_func(image=im, k_ind=k_ind, k_value=k_val, image_ind=im_ind, threshold=th, roi_cent=roi_c,
#                       roi_dim=roi_d, cal=cal, kernel=kern, bpp=16, normalize=norm, enabled=en)
    t.add_work_item(image=im, k_ind=k_ind, k_value=k_val, image_ind=im_ind, threshold=th, roi_cent=roi_c,
                    roi_dim=roi_d, cal=cal, kernel=kern, bpp=16, normalize=norm, enabled=en)

    time.sleep(1.0)

    res = t.get_result(wait=False)
    logger.info("RESULT from task: {0}, {1}".format(res, type(res)))

    t.stop_processes()
    t.cancel()
