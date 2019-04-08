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
logger.setLevel(logging.INFO)


from collections import namedtuple


JobStruct = namedtuple("JobStruct", "image k_ind k_value image_ind threshold roi_cent roi_dim "
                                    "cal kernel bpp normalize enabled")
"""
QuadImage stores one full image from a quad scan.
:param k_ind: index for the k value list for this image
:param k_value: k value for this image
:param image_ind: index for the image list for this image
:param image: image data in the form of a 2d numpy array
"""


var_dict = dict()


def init_worker(sh_mem_image, sh_mem_roi, image_shape):
    var_dict["image"] = sh_mem_image
    var_dict["image_shape"] = image_shape
    var_dict["roi"] = sh_mem_roi


def work_func_shared(image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False):
    t0 = time.time()
    shape = var_dict["image_shape"]
    logger.info("Processing image {0} in pool".format(image_ind))
    # print("Processing image {0} in pool".format(image_ind))
    # print("Processing image {0} in pool, size {1}".format(image_ind, shape))
    image = np.frombuffer(var_dict["image"], "i", shape[0]*shape[1],
                          shape[0] * shape[1] * image_ind * np.dtype("i").itemsize).reshape((shape[0], shape[1]))
    # roi = np.frombuffer(var_dict["roi"], "f", shape[0]*shape[1],
    #                     image_ind*np.dtype("f").itemsize).reshape((shape[0], shape[1]))
    roi = np.frombuffer(var_dict["roi"], "f", roi_dim[0] * roi_dim[1],
                        shape[0] * shape[1] * image_ind * np.dtype("f").itemsize).reshape(roi_dim)

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

    # print("Medfilt done")
    # Threshold image
    try:
        if threshold is None:
            threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
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

    # print("Process time {0:.2f} ms".format((time.time()-t0)*1e3))

    return image_ind, x_cent, sigma_x, y_cent, sigma_y, q, enabled
    # return image_ind, 0, 0, 0, 0, 0, 0


def work_func(image, image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3, bpp=16, normalize=False):
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

    # print("Medfilt done")
    # Threshold image
    try:
        if threshold is None:
            threshold = pic_roi[0:20, 0:20].mean() * 3 + pic_roi[-20:, -20:].mean() * 3
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

    return pic_roi, line_x, line_y, x_cent, sigma_x, y_cent, sigma_y, q, enabled


def work_func_shared2(image_ind):
    t0 = time.time()
    print("Processing image {0}".format(image_ind))

    return image_ind


class ProcessPoolTaskShared(Task):
    """
    Start up a process pool that can consume data. Emit trigger when queue is empty.

    The results of the tasks are stored in a list.
    """

    def __init__(self, image_shape, number_processes=multiprocessing.cpu_count(), name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.work_func = work_func_shared
        self.lock = multiprocessing.Lock()
        self.finish_process_event = threading.Event()
        self.result_ready_event = threading.Event()
        self.pending_work_items = list()

        # self.sh_mem_lock_list = list()      # List of locks that handle access to shared memory arrays
        # self.job_launcher_thread = None     # Thread that waits for shared memory to be available and
                                            # copies image data for a new job to that location and
                                            # submits to the process pool
        self.sh_mem_rawarray = None         # Shared memory array
        self.sh_mem_roi_rawarray = None
        self.sh_np_array = None             # Numpy array from the shared buffer. Shape [im_x, im_y, n_proc]
        self.sh_roi_np_array = None
        self.image_shape = image_shape
        self.job_queue = None
        self.current_job_list = None
        self.mem_ind_queue = None

        self.num_processes = number_processes
        self.pool = None
        self.next_process_id = 0
        self.completed_work_items = 0
        self.id_lock = threading.Lock()

        self.result_thread = None
        self.stop_launcher_thread_flag = False
        self.result_dict = dict()

        self.logger.setLevel(logging.INFO)

    def run(self):
        self.create_processes()
        Task.run(self)      # self.action is called here

    def start(self):
        self.next_process_id = 0
        Task.start(self)

    def action(self):
        self.logger.info("{0}: starting processing pool for "
                         "executing function {1} across {2} processes.".format(self, self.work_func, self.num_processes))
        self.logger.info("{1}: Starting {0} processes".format(self.num_processes, self))
        self.completed_work_items = 0
        self.stop_launcher_thread_flag = False

        # self.job_launcher_thread = threading.Thread(target=self.job_launcher)
        # self.job_launcher_thread.start()

        self.finish_process_event.wait(self.timeout)
        if self.finish_process_event.is_set() is False:
            self.cancel()
            return
        self.logger.debug("Finish process event set")

        t0 = time.time()
        while self.completed_work_items < self.next_process_id:
            time.sleep(0.01)
            if time.time() - t0 > 5.0:
                self.logger.error("{0}: Timeout waiting for {1} work items to complete".format(self, self.next_process_id - self.completed_work_items))
                self.stop_processes(terminate=True)
                # self.result = self.result_dict
                return
        self.stop_processes(terminate=False)
        self.result = self.result_dict
        self.pool = None

    def create_processes(self):
        if self.pool is not None:
            self.pool.terminate()
        self.logger.info("{1}: Creating {0} processes".format(self.num_processes, self))
        n_mem = self.num_processes
        self.sh_mem_rawarray = multiprocessing.RawArray("i", self.image_shape[0]*self.image_shape[1]*n_mem)
        self.sh_np_array = np.frombuffer(self.sh_mem_rawarray, dtype="i").reshape((self.image_shape[0],
                                                                                   self.image_shape[1],
                                                                                   n_mem))
        self.sh_mem_roi_rawarray = multiprocessing.RawArray("f", self.image_shape[0]*self.image_shape[1]*n_mem)
        # self.sh_roi_np_array = np.frombuffer(self.sh_mem_roi_rawarray, dtype="f").reshape((self.image_shape[0],
        #                                                                                    self.image_shape[1],
        #                                                                                    n_mem))

        self.pool = multiprocessing.Pool(self.num_processes, initializer=init_worker,
                                         initargs=(self.sh_mem_rawarray, self.sh_mem_roi_rawarray,
                                                   (self.image_shape[0], self.image_shape[1], n_mem)))

        self.job_queue = Queue.Queue()
        self.mem_ind_queue = Queue.Queue()

        [self.mem_ind_queue.put(x) for x in range(n_mem)]       # Fill queue with available memory indices
        self.current_job_list = [None for x in range(n_mem)]

    def add_work_item(self, image, k_ind, k_value, image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3,
                       bpp=16, normalize=False, enabled=True):
        self.logger.debug("{0}: Adding work item".format(self))
        # self.logger.debug("{0}: Args: {1}, kwArgs: {2}".format(self, args, kwargs))
        if not self.finish_process_event.is_set():
            proc_id = self.next_process_id
            self.next_process_id += 1
            job = JobStruct(image=image, k_ind=k_ind, k_value=k_value, image_ind=image_ind, threshold=threshold,
                            roi_cent=roi_cent, roi_dim=roi_dim, cal=cal, kernel=kernel, bpp=bpp, normalize=normalize,
                            enabled=enabled)
            self.job_queue.put(job)
            self.logger.debug("{0}: Work item added to queue. "
                              "Process id: {1}, job queue length: {2}".format(self, proc_id, self.job_queue.qsize()))
            if self.mem_ind_queue.qsize() > 0:
                self.job_launch()

    def job_launcher(self):
        self.logger.info("{0} Entering job launcher thread".format(self))
        while not self.stop_launcher_thread_flag:

            # wait for a job to be issued:
            try:
                job = self.job_queue.get(False, 0.05)    # type: JobStruct
            except Queue.Empty:
                # If the queue was empty, timeout and check the stop_result_flag again
                continue
            self.logger.info("{0} Job received: {1} {2}".format(self, job.k_ind, job.image_ind))

            # wait for memory to be available:
            while not self.stop_launcher_thread_flag:
                try:
                    ind = self.mem_ind_queue.get(False, 0.05)
                except Queue.Empty:
                    # If the queue was empty, timeout and check the stop_result_flag again
                    continue

                # copy image data to shared memory:
                np.copyto(self.sh_np_array[:, :, ind], job.image)
                kwargs = {"image_ind": ind, "threshold": job.threshold, "roi_cent": job.roi_cent,
                          "roi_dim": job.roi_dim, "cal": job.cal, "kernel": job.kernel,
                          "bpp": job.bpp, "normalize": job.normalize}

                # Save job in list:
                self.current_job_list[ind] = job

                # Start processing:
                self.logger.info("{0} Putting job on pool process {1}".format(self, ind))
                if not self.stop_launcher_thread_flag:
                    self.logger.debug("{0} apply async {1}".format(self, self.work_func))
                    self.pool.apply_async(self.work_func, kwds=kwargs, callback=self.pool_callback)
                    # self.pool.apply_async(self.work_func, args=(ind, ), callback=self.pool_callback)
                break

    def job_launch(self):
        try:
            job = self.job_queue.get(False, 0.05)  # type: JobStruct
        except Queue.Empty:
            # If the queue was empty, timeout and check the stop_result_flag again
            return
        self.logger.debug("{0} Job received: {1} {2}".format(self, job.k_ind, job.image_ind))

        # wait for memory to be available:
        while not self.stop_launcher_thread_flag:
            try:
                ind = self.mem_ind_queue.get(False, 0.05)
            except Queue.Empty:
                # If the queue was empty, timeout and check the stop_result_flag again
                continue

            self.logger.debug("{0} Copy data: {1}".format(self, job.k_ind))

            # copy image data to shared memory:
            np.copyto(self.sh_np_array[:, :, ind], job.image)
            self.logger.debug("{0} Copy data done {1}".format(self, job.k_ind))
            kwargs = {"image_ind": ind, "threshold": job.threshold, "roi_cent": job.roi_cent,
                      "roi_dim": job.roi_dim, "cal": job.cal, "kernel": job.kernel,
                      "bpp": job.bpp, "normalize": job.normalize}

            # Save job in list:
            self.current_job_list[ind] = job

            # Start processing:
            self.logger.debug("{0} Putting job {1} on pool process {2}".format(self, job.k_ind, ind))
            if not self.stop_launcher_thread_flag:
                self.logger.debug("{0} apply async {1}".format(self, self.work_func))
                self.pool.apply_async(self.work_func, kwds=kwargs, callback=self.pool_callback)
                # self.pool.apply_async(self.work_func, args=(ind, ), callback=self.pool_callback)
            return

    def pool_callback(self, result):
        self.logger.debug("Pool callback returned {0}".format(result[0]))
        self.completed_work_items += 1

        # Cancel task if an exception was received:
        if isinstance(result, Exception):
            self.logger.exception("{0} pool callback exception: {1}".format(self, result))
            self.cancel()
            return

        ind = result[0]
        x_cent = result[1]
        sigma_x = result[2]
        y_cent = result[3]
        sigma_y = result[4]
        q = result[5]
        enabled = result[6]
        job = self.current_job_list[ind]    # type: JobStruct

        pic_roi = np.frombuffer(self.sh_mem_roi_rawarray, "f", job.roi_dim[0] * job.roi_dim[1], self.image_shape[0] *
                                self.image_shape[1] * ind * np.dtype("f").itemsize).reshape(job.roi_dim)
        line_x = pic_roi.sum(0)
        line_y = pic_roi.sum(1)
        ProcessedImage(k_ind=job.k_ind, k_value=job.k_value, image_ind=job.image_ind, pic_roi=pic_roi,
                       line_x=line_x, line_y=line_y, x_cent=x_cent, sigma_x=sigma_x, y_cent=y_cent, sigma_y=sigma_y,
                       q=q, enabled=enabled, threshold=job.threshold)

        # Mark this index as free for processing:
        self.mem_ind_queue.put(ind)
        self.logger.info("{0} index {1} available for job".format(self, ind))
        self.job_launch()

        # Signal that a result is ready:
        self.result_ready_event.set()
        if self.completed_work_items < self.next_process_id:
            self.result_ready_event.clear()

        for callback in self.callback_list:
            callback(self)

    def stop_processes(self, terminate=True):
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

    def cancel(self):
        self.finish_processing()
        Task.cancel(self)

    def get_result(self, wait, timeout=-1):
        if self.completed is not True:
            if wait is True:
                if timeout > 0:
                    self.result_ready_event.wait(timeout)
                else:
                    self.result_ready_event.wait()
        return self.result

    def get_remaining_workitems(self):
        return self.next_process_id-self.completed_work_items

    def emit(self):
        self.result_ready_event.set()
        Task.emit(self)


def callback_f(task):
    name = task.get_name()
    res = task.get_result(wait=False)
    logger.debug("Callback from task {0}: {1}".format(name, res))


if __name__ == "__main__":

    im = np.random.random((1280, 1024))
    im_list = [np.random.randint(0, 16383, (1280, 1024)) for k in range(16)]

    t = ProcessPoolTaskShared(image_shape=im.shape, name="test", callback_list=[callback_f])
    t.start()
    time.sleep(1.0)

#    im = np.random.random((1000, 1000))
    k_ind = 0
    k_val = 0
    im_ind = 0
    th = 0
    roi_c = [256, 150]
    roi_d = [256, 256]
    cal = [1, 1]
    kern = 3
    norm = False
    en = True

    qi = QuadImage(k_ind, k_val, im_ind, im)
#    process_image_func(image=im, k_ind=k_ind, k_value=k_val, image_ind=im_ind, threshold=th, roi_cent=roi_c,
#                       roi_dim=roi_d, cal=cal, kernel=kern, bpp=16, normalize=norm, enabled=en)
#     t.add_work_item(image=im, k_ind=k_ind, k_value=k_val, image_ind=im_ind, threshold=th, roi_cent=roi_c,
#                     roi_dim=roi_d, cal=cal, kernel=kern, bpp=16, normalize=norm, enabled=en)

    for im in im_list:
        t.add_work_item(image=im, k_ind=k_ind, k_value=k_val, image_ind=im_ind, threshold=th, roi_cent=roi_c,
                        roi_dim=roi_d, cal=cal, kernel=kern, bpp=16, normalize=norm, enabled=en)
        k_ind += 1

    t0 = time.time()
    logger.info("WAITING")
    while t.get_remaining_workitems() > 0:
        res = t.get_result(wait=True, timeout=1.0)
        dt = time.time() - t0
        if dt > 2:
            break
    logger.info("RESULT from task: {0}, {1:.2f} ms".format(res, dt * 1e3))

    time.sleep(1.0)
    t.finish_processing()
    t.get_done_event().wait()

    logger.info("=========================================================================")
    logger.info("TOTAL time: {0:.2f} ms".format((time.time() - t0) * 1e3))
    # t.cancel()



