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
import os
from collections import namedtuple
import pprint
import traceback
from scipy.signal import medfilt2d
from tasks.GenericTasks import *

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


QuadImage = namedtuple("QuadImage", "k_ind k_value image_ind image")
ProcessedImage = namedtuple("ProcessedImage",
                            "k_ind image_ind pic_roi line_x line_y x_cent y_cent sigma_x sigma_y q enabled")
#
#
# # From: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
# def _async_raise(tid, exctype):
#     """Raises an exception in the threads with id tid"""
#     if not inspect.isclass(exctype):
#         raise TypeError("Only types can be raised (not instances)")
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
#                                                      ctypes.py_object(exctype))
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         # "if it returns a number greater than one, you're in trouble,
#         # and you should call it again with exc=NULL to revert the effect"
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")
#
#
# class ThreadWithExc(threading.Thread):
#     """A thread class that supports raising exception in the thread from
#        another thread.
#     """
#     def _get_my_tid(self):
#         """determines this (self's) thread id
#
#         CAREFUL : this function is executed in the context of the caller
#         thread, to get the identity of the thread represented by this
#         instance.
#         """
#         if not self.isAlive():
#             raise threading.ThreadError("the thread is not active")
#
#         # do we have it cached?
#         if hasattr(self, "_thread_id"):
#             return self._thread_id
#
#         # no, look for it in the _active dict
#         for tid, tobj in threading._active.items():
#             if tobj is self:
#                 self._thread_id = tid
#                 return tid
#
#         # TODO: in python 2.6, there's a simpler way to do : self.ident
#
#         raise AssertionError("could not determine the thread's id")
#
#     def raise_exc(self, exctype):
#         """Raises the given exception type in the context of this thread.
#
#         If the thread is busy in a system call (time.sleep(),
#         socket.accept(), ...), the exception is simply ignored.
#
#         If you are sure that your exception should terminate the thread,
#         one way to ensure that it works is:
#
#             t = ThreadWithExc( ... )
#             ...
#             t.raise_exc( SomeException )
#             while t.isAlive():
#                 time.sleep( 0.1 )
#                 t.raise_exc( SomeException )
#
#         If the exception is to be caught by the thread, you need a way to
#         check that your thread has caught it.
#
#         CAREFUL : this function is executed in the context of the
#         caller thread, to raise an exception in the context of the
#         thread represented by this instance.
#         """
#         _async_raise(self._get_my_tid(), exctype)
#
#

#
#
# class Task(object):
#     class CancelException(Exception):
#         pass
#
#     def __init__(self, name=None, action_exec_type="thread", timeout=None, trigger_dict=dict(), callback_list=list()):
#         """
#
#         :param name: Optional name to identify the task
#         :param action_exec_type: "thread" or "process" Determines if action method is executed in thread or process
#         :param timeout: timeout in seconds that is spent waiting for triggers
#         :param trigger_dict: Dict of tasks that need to trigger before executing the action method of this task
#         :param callback_list:
#         """
#         self.id = uuid.uuid1()
#         if name is None:
#             name = str(self.id)
#         self.name = name
#         self.action_exec_type = action_exec_type
#         self.trigger_dict = dict()
#         self.trigger_done_list = list()
#         self.trigger_result_dict = dict()
#         self.lock = threading.Lock()
#         self.event_done = threading.Event()
#         self.trigger_event = threading.Event()
#         self.started = False
#         self.completed = False
#         self.cancelled = False
#         self.result = None
#         self.timeout = timeout
#         self.run_thread = None
#         self.callback_list = callback_list
#
#         self.logger = logging.getLogger("Task.{0}".format(self.name.upper()))
#         self.logger.setLevel(logging.DEBUG)
#
#         for e in trigger_dict.values():
#             self.add_trigger(e)
#
#     def add_trigger(self, trigger_task):
#         self.logger.info("{0} adding trigger {1}".format(self, trigger_task.name))
#         with self.lock:
#             self.trigger_dict[trigger_task.id] = trigger_task
#
#     def add_callback(self, callback):
#         self.logger.info("{0} adding callback {1}".format(self, callback))
#         with self.lock:
#             self.callback_list.append(callback)
#
#     def get_done_event(self):
#         return self.event_done
#
#     def get_signal_event(self):
#         return self.signal_event
#
#     def get_result(self, wait, timeout=-1):
#         if self.completed is not True:
#             if wait is True:
#                 if timeout > 0:
#                     self.event_done.wait(timeout)
#                 else:
#                     self.event_done.wait()
#         return self.result
#
#     def action(self):
#         self.logger.info("{0} entering action.".format(self))
#         self.result = None
#
#     def emit(self):
#         self.logger.debug("{0} done. Emitting signal".format(self))
#
#         self.event_done.set()
#         self.logger.debug("{0}: Calling {1} callbacks".format(self, len(self.callback_list)))
#         for callback in self.callback_list:
#             callback(self)
#
#     def run(self):
#         self.logger.debug("{0} entering run.".format(self))
#
#         # Setup threads for triggers:
#         already_done_flag = True
#         for tr_id in self.trigger_dict:
#             already_done_flag = False
#             th = threading.Thread(target=self._wait_trigger, args=(self.trigger_dict[tr_id],))
#             th.start()
#         if already_done_flag is False:
#             # Wait for trigger_event:
#             if self.timeout is not None:
#                 completed_flag = self.trigger_event.wait(self.timeout)
#                 if completed_flag is False:
#                     self.logger.info("{0} {1} timed out.".format(type(self), self.name))
#                     self.cancel()
#                     return
#             else:
#                 self.trigger_event.wait()
#
#         if self.is_cancelled() is True:
#             return
#
#             self.logger.debug("{0} triggers ready".format(self))
#         try:
#             if self.action_exec_type == "thread":
#                 self.action()
#                 self.logger.debug("{0} action done".format(self))
#             else:
#                 process = multiprocessing.Process(target=self.action)
#                 process.start()
#                 self.logger.debug("{0} action process done".format(self))
#         except self.CancelException:
#             self.logger.info("{0} Cancelled".format(self))
#             return
#         except Exception as e:
#             self.logger.error("{0} exception: {1}".format(self, traceback.format_exc()))
#             self.result = e
#             self.result = traceback.format_exc()
#             self.cancel()
#             return
#         with self.lock:
#             self.completed = True
#         self.emit()
#
#     def start(self):
#         self.logger.debug("{0} starting.".format(self))
#         with self.lock:
#             self.trigger_done_list = list()
#             self.trigger_result_dict = dict()
#             self.event_done.clear()
#             self.trigger_event.clear()
#             self.started = True
#             self.completed = False
#             self.cancelled = False
#             self.result = None
#         # self.run_thread = threading.Thread(target=self.run)
#         self.run_thread = ThreadWithExc(target=self.run)
#         self.run_thread.start()
#
#     def cancel(self):
#         self.logger.debug("{0} cancelling.".format(self))
#         with self.lock:
#             self.started = False
#             self.completed = False
#             self.cancelled = True
#             self.trigger_done_list = list()
#             self.trigger_result_dict = dict()
#             self.trigger_event.set()
#             self.run_thread.raise_exc(self.CancelException)
#         self.emit()
#
#     def is_cancelled(self):
#         return self.cancelled
#
#     def is_done(self):
#         return self.completed
#
#     def is_started(self):
#         return self.started
#
#     def _wait_trigger(self, trigger_task):
#         self.logger.debug("{0} starting wait for task {1}.".format(self, trigger_task.name))
#         e = trigger_task.get_done_event()
#         e.wait()
#         # Cancel task if trigger was cancelled:
#         if trigger_task.is_cancelled() is True:
#             self.cancel()
#             return
#         with self.lock:
#             self.trigger_done_list.append(trigger_task.id)
#             if len(self.trigger_done_list) == len(self.trigger_dict):
#                 self.trigger_event.set()
#
#     def __eq__(self, other):
#         """
#         Check if this task is the same as another based on the id, name or object
#         :param other: Other task id number, name, or task object
#         :return: True if it is the same
#         """
#         if type(other) == uuid.UUID:
#             eq = self.id == other
#         elif type(other) == str:
#             eq = self.name == other
#         elif type(other) == Task:
#             eq = self.id == other.id
#         else:
#             eq = False
#         return eq
#
#     def __repr__(self):
#         s = "{0} {1}".format(type(self), self.name)
#         return s
#
#     def __str__(self):
#         s = "{0} {1}".format(type(self).__name__, self.name)
#         return s
#
#
# class DelayTask(Task):
#     def __init__(self, delay, name=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.delay = delay
#
#     def action(self):
#         self.logger.info("{0} Starting delay of {1} s".format(self, self.delay))
#         time.sleep(self.delay)
#         self.result = True
#
#
# class CallableTask(Task):
#     def __init__(self, call_func, call_args=list(), call_kwargs=dict(),
#                  name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.callable = call_func
#         self.call_args = call_args
#         self.call_kwargs = call_kwargs
#
#         self.logger.setLevel(logging.WARN)
#
#     def action(self):
#         self.logger.info("{0} calling {1}. ".format(self, self.callable))
#         # Exceptions are caught in the parent run thread.
#         # Still, I want to log an error message and re-raise
#         try:
#             res = self.callable(*self.call_args, **self.call_kwargs)
#         except Exception as e:
#             self.logger.error("Error when executing {0} with args {1}, {2}:\n {3}".format(self.callable,
#                                                                                           self.call_args,
#                                                                                           self.call_kwargs,
#                                                                                           e))
#             raise e
#         self.result = res
#
#
# class RepeatTask(Task):
#     """
#     Repeat a task a number of times and store the last result.
#     If intermediate results are needed: wait for event_done emission
#     from the repeating task. Use repetitions <= 0 for infinite number of
#     repeats.
#     """
#     def __init__(self, task, repetitions, delay=0, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.task = task
#         self.repetitions = repetitions
#         self.delay = delay
#
#     def action(self):
#         self.logger.info("{0} repeating task {1} {2} times.".format(self, self.task, self.repetitions))
#         current_rep = 0
#         result_list = list()
#         while current_rep < self.repetitions:
#             self.task.start()
#             res = self.task.get_result(wait=True, timeout=self.timeout)
#             # Check if cancelled or error..
#             if self.task.is_cancelled() is True:
#                 self.cancel()
#                 break
#             if self.is_cancelled() is True:
#                 break
#             if self.repetitions >= 0:
#                 # Only store intermediate results in a list if there is not an infinite number of repetitions.
#                 result_list.append(res)
#             time.sleep(self.delay)
#             current_rep += 1
#         self.result = result_list
#
#
# class SequenceTask(Task):
#     """
#     Run a sequence of tasks in a list and emit done event after all
#     are completed.
#
#     The results of the tasks are stored in a list.
#     """
#     def __init__(self, task_list, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.task_list = task_list
#
#     def action(self):
#         self.logger.info("{0} running task sequence of length {1}.".format(self, len(self.task_list)))
#         res = list()
#
#         for t in self.task_list:
#             t.start()
#             res.append(t.get_result(wait=True))
#             # Check if cancelled or error..
#             if t.is_cancelled() is True:
#                 self.cancel()
#                 break
#             if self.is_cancelled() is True:
#                 break
#
#         self.result = res
#
#
# class BagOfTasksTask(Task):
#     """
#     Run a list of tasks and emit done event after all
#     are completed.
#
#     The results of the tasks are stored in a list.
#     """
#     def __init__(self, task_list, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.task_list = task_list
#
#     def action(self):
#         self.logger.info("{0} running {1} tasks.".format(self, len(self.task_list)))
#         res = list()
#
#         for t in self.task_list:
#             t.start()
#         for t in self.task_list:
#             # Task results are stored in a list in the order they were created.
#             res.append(t.get_result(wait=True))
#             # Check if cancelled or error..
#             if t.is_cancelled() is True:
#                 self.cancel()
#                 break
#             if self.is_cancelled() is True:
#                 break
#
#         self.result = res
#
#
# class MapToProcessTask(Task):
#     """
#     Map a number of tasks to pool of processes.
#
#     The results of the tasks are stored in a list.
#     """
#
#     def __init__(self, f, data_list, number_processes=5, name=None, trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.f = f
#         self.data_list = data_list
#         self.lock = multiprocessing.Lock()
#         self.num_proc = number_processes
#         self.pool
#
#     def action(self):
#         self.logger.info("{0} executing function on data of length {1} across {2} processes.".format(self,
#                                                                                                      len(self.task_list),
#                                                                                                      self.num_proc))
#         res = list()
#         pool = multiprocessing.Pool(processes=self.num_proc)
#         with self.lock:
#             res = pool.map(self.f, self.data_list)
#
#         self.result = res
#
#     def add_data(self, data_list):
#         with self.lock:
#             self.data_list.append(data_list)
#
#     def set_data_list(self, data_list):
#         with self.lock:
#             self.data_list = data_list
#
#
# def clearable_pool_worker(in_queue, out_queue):
#     while True:
#         task = in_queue.get()
#         f = task[0]
#         args = task[1]
#         kwargs = task[2]
#         proc_id = task[3]
#         # logger.debug("Pool worker executing {0} with args {1}, {2}".format(f, args, kwargs))
#         # logger.debug("Pool worker executing {0} ".format(f))
#         try:
#             retval = f(*args, **kwargs)
#             # logger.debug("Putting {0} on out_queue".format(retval))
#             # logger.debug("Putting {0} result on out_queue".format(f))
#         except Exception as e:
#             retval = e
#             logger.error("{0} Error {1} ".format(f, retval))
#         out_queue.put((retval, proc_id))
#
#
# class ProcessPoolTask(Task):
#     """
#     Start up a process pool that can consume data. Emit trigger when queue is empty.
#
#     The results of the tasks are stored in a list.
#     """
#
#     def __init__(self, work_func, number_processes=multiprocessing.cpu_count(), name=None, timeout=None,
#                  trigger_dict=dict(), callback_list=list()):
#         Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
#         self.work_func = work_func
#         self.lock = multiprocessing.Lock()
#         self.finish_process_event = threading.Event()
#         self.pending_work_items = list()
#
#         self.in_queue = multiprocessing.Queue()
#         self.out_queue = multiprocessing.Queue()
#         self.num_processes = number_processes
#         self.processes = None
#         self.next_process_id = 0
#         self.completed_work_items = 0
#         self.id_lock = threading.Lock()
#
#         self.result_thread = None
#         self.stop_result_thread_flag = False
#         self.result_dict = dict()
#
#         self.logger.setLevel(logging.INFO)
#
#     def run(self):
#         self.create_processes()
#         Task.run(self)
#
#     def start(self):
#         self.next_process_id = 0
#         Task.start(self)
#
#     def action(self):
#         self.logger.info("{0}: starting processing pool for "
#                          "executing function {1} across {2} processes.".format(self, self.work_func, self.num_processes))
#         self.logger.info("{1}: Starting {0} processes".format(self.num_processes, self))
#         self.completed_work_items = 0
#         for p in self.processes:
#             p.start()
#         self.stop_result_thread_flag = False
#         self.result_thread = threading.Thread(target=self.result_thread_func)
#         self.result_thread.start()
#
#         self.finish_process_event.wait(self.timeout)
#         if self.finish_process_event.is_set() is False:
#             self.cancel()
#         while self.completed_work_items < self.next_process_id:
#             time.sleep(0.01)
#         self.stop_processes()
#         self.result = self.result_dict
#
#     def add_work_item(self, *args, **kwargs):
#         self.logger.info("{0}: Adding work item".format(self))
#         # self.logger.debug("{0}: Args: {1}, kwArgs: {2}".format(self, args, kwargs))
#         proc_id = self.next_process_id
#         self.next_process_id += 1
#         self.in_queue.put((self.work_func, args, kwargs, proc_id))
#         self.result_dict[proc_id] = None
#         self.logger.debug("{0}: Queue added. Process id: {1}".format(self, proc_id))
#
#     def create_processes(self):
#         if self.processes is not None:
#             self.stop_processes()
#
#         self.logger.info("{1}: Creating {0} processes".format(self.num_processes, self))
#         p_list = list()
#         for p in range(self.num_processes):
#             p = multiprocessing.Process(target=clearable_pool_worker, args=(self.in_queue, self.out_queue))
#             p_list.append(p)
#         self.processes = p_list
#
#     def stop_processes(self):
#         self.logger.info("{0}: Stopping processes".format(self))
#         if self.processes is not None:
#             for p in self.processes:
#                 p.terminate()
#         # self.processes = None
#         self.stop_result_thread_flag = True
#         try:
#             self.result_thread.join(1.0)
#         except AttributeError:
#             pass
#         self.result_thread = None
#
#     def finish_processing(self):
#         self.finish_process_event.set()
#
#     def cancel(self):
#         self.finish_processing()
#         Task.cancel(self)
#
#     def clear_pending_tasks(self):
#         self.logger.info("{0}: Clearing pending tasks".format(self))
#         while self.in_queue.empty() is False:
#             try:
#                 self.logger.debug("get no wait")
#                 work_item = self.in_queue.get_nowait()
#                 proc_id = work_item[3]
#                 self.logger.debug("Removing task {0}".format(id))
#                 try:
#                     self.result_dict.pop(proc_id)
#                 except KeyError:
#                     pass
#             except multiprocessing.queues.Empty:
#                 self.logger.debug("In-queue empty")
#                 break
#
#     def result_thread_func(self):
#         self.logger.debug("{0}: Starting result collection thread".format(self))
#         while self.stop_result_thread_flag is False:
#             try:
#                 result = self.out_queue.get(True, 0.1)
#             except multiprocessing.queues.Empty:
#                 continue
#             retval = result[0]
#             proc_id = result[1]
#             self.completed_work_items += 1
#             try:
#                 self.result_dict[proc_id] = retval
#             except KeyError as e:
#                 self.logger.error("Work item id {0} not found.".format(proc_id))
#                 raise e
#             if isinstance(retval, Exception):
#                 raise retval
#             self.logger.debug("{0}: result_dict {1}".format(self, pprint.pformat(self.result_dict)))
#             self.result = retval
#             for callback in self.callback_list:
#                 callback(self)


class TangoDeviceConnectTask(Task):
    def __init__(self, device_name, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
        # Exceptions are caught in the parent run thread.
        self.logger.debug("Connecting to {0}".format(self.device_name))
        dev = pt.DeviceProxy(self.device_name)
        self.result = dev


class TangoReadAttributeTask(Task):
    def __init__(self, attribute_name, device_name, device_handler, name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler

        self.logger.setLevel(logging.WARN)

    def action(self):
        self.logger.info("{0} reading {1} on {2}. ".format(self, self.attribute_name, self.device_name))
        dev = self.device_handler.get_device(self.device_name)
        attr = dev.read_attribute(self.attribute_name)
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
        self.logger.info("{0} writing {1} to {2} on {2}. ".format(self,
                                                                  self.value,
                                                                  self.attribute_name,
                                                                  self.device_name))
        dev = self.device_handler.get_device(self.device_name)
        res = dev.write_attribute(self.attribute_name, self.value)
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
                self.cancel()
                break
            t1 = time.time()
            wait_time = self.interval - (t1 - t0)

        self.result = read_task.get_result(wait=False)


class LoadImageTask(Task):
    def __init__(self, image_name, path=".", name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.image_name = image_name
        self.path = path
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
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


class LoadQuadScanDirTask(Task):
    def __init__(self, quadscandir, process_now=True, threshold=0.0, kernel_size=3, image_processor_task=None,
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.INFO)

        self.pathname = quadscandir
        self.process_now = process_now
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.processed_image_list = list()
        self.task_seq = None
        if image_processor_task is None:
            self.image_processor = ImageProcessorTask(threshold=threshold, kernel=kernel_size,
                                                      trigger_dict=trigger_dict, name="loaddir_image_proc")
        else:
            # If the image_processor was supplied, don't add self as trigger.
            self.image_processor = image_processor_task
        if self.image_processor.is_started() is False:
            self.logger.info("Starting image_processor")
            self.image_processor.start()
        self.image_processor.add_callback(self.processed_image_done)

    def action(self):
        load_dir = self.pathname
        self.logger.debug("{1}: Loading from {0}".format(load_dir, self))
        try:
            os.listdir(load_dir)
        except OSError as e:
            e = "List dir failed: {0}".format(e)
            self.result = e
            self.cancel()

        # See if there is a file called daq_info.txt
        filename = "daq_info.txt"
        if os.path.isfile(os.path.join(load_dir, filename)) is False:
            e = "daq_info.txt not found in {0}".format(load_dir)
            self.result = e
            self.cancel()

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

        file_list = os.listdir(load_dir)
        image_file_list = list()
        load_task_list = list()
        for file_name in file_list:
            if file_name.endswith(".png"):
                image_file_list.append(file_name)
                t = LoadImageTask(file_name, load_dir, name=file_name,
                                  callback_list=[self.image_processor.process_image])
                t.logger.setLevel(logging.WARN)
                load_task_list.append(t)

        self.logger.debug("{1} Found {0} images in directory".format(len(image_file_list), self))
        self.task_seq = SequenceTask(load_task_list, name="load_seq")
        self.task_seq.start()
        # Wait for image sequence to be done reading:
        image_list = self.task_seq.get_result(wait=True)
        # Now wait for images to be done processing:
        self.image_processor.stop_processing()
        self.image_processor.get_result(wait=True)
        ImageList = namedtuple("ImageList", "daq_data images proc_images")
        self.result = ImageList(data_dict, image_list, self.processed_image_list)

    def processed_image_done(self, image_processor_task):
        # type: (ImageProcessorTask) -> None
        proc_image = image_processor_task.get_result(wait=False)    # type: ProcessedImage
        if image_processor_task.is_done() is False:
            self.logger.debug("Adding processed image {0} {1} to list".format(proc_image.k_ind, proc_image.image_ind))
            self.processed_image_list.append(proc_image)

    def cancel(self):
        if self.task_seq is not None:
            self.task_seq.cancel()
        Task.cancel(self)


def process_image_func(image, k_ind, image_ind, threshold, roi_cent, roi_dim, cal=[1.0, 1.0], kernel=3, bpp=16):
    logger.info("Processing image {0}, {1} in pool".format(k_ind, image_ind))
    x = np.array([int(roi_cent[0] - roi_dim[0] / 2.0), int(roi_cent[0] + roi_dim[0] / 2.0)])
    y = np.array([int(roi_cent[1] - roi_dim[1] / 2.0), int(roi_cent[1] + roi_dim[1] / 2.0)])

    # Extract ROI and convert to double:
    pic_roi = np.double(image[x[0]:x[1], y[0]:y[1]])
    # logger.debug("pic_roi size: {0}".format(pic_roi.shape))
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
    result = ProcessedImage(k_ind=k_ind, image_ind=image_ind, pic_roi=pic_roi, line_x=line_x, line_y=line_y,
                            x_cent=x_cent, y_cent=y_cent, sigma_x=sigma_x, sigma_y=sigma_y, q=q, enabled=enabled)

    return result


class ImageProcessorTask(Task):
    def __init__(self, roi_cent=None, roi_dim=None, threshold=0.1, cal=[1.0, 1.0],
                 kernel=3, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self,  name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.logger.setLevel(logging.INFO)

        self.kernel = kernel
        self.cal = cal
        self.threshold = threshold
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim
        self.processor = ProcessPoolTask(process_image_func, name="process_pool", callback_list=[self._image_done])
        # self.processor = ProcessPoolTask(test_f, name="process_pool")
        self.stop_processing_event = threading.Event()

    def action(self):
        self.logger.info("{0} entering action. ".format(self))
        self.processor.start()
        self.stop_processing_event.wait(self.timeout)
        if self.stop_processing_event.is_set() is False:
            self.cancel()
            return
        self.logger.info("{0} exit processing")
        self.processor.finish_processing()
        self.processor.get_result(wait=True, timeout=self.timeout)
        self.result = True

    def process_image(self, task, bpp=16):
        quad_image = task.get_result(wait=False)    # type: QuadImage
        self.logger.info("{0}: Adding image {1} {2} to processing queue".format(self, quad_image.k_ind,
                                                                                quad_image.image_ind))
        if self.roi_dim is None:
            roi_dim = quad_image.image.shape
        else:
            roi_dim = self.roi_dim
        if self.roi_cent is None:
            roi_cent = [roi_dim[0]/2, roi_dim[1]/2]
        else:
            roi_cent = self.roi_cent
        # self.processor.add_work_item("apa")
        self.processor.add_work_item(quad_image.image, quad_image.k_ind, quad_image.image_ind, self.threshold,
                                     roi_cent, roi_dim, self.cal, self.kernel, bpp)

    def _image_done(self, processor_task):
        # type: (ProcessPoolTask) -> None
        if self.is_done() is False:
            self.logger.debug("{0}: Image processing done.".format(self))
            self.result = processor_task.get_result(wait=False)
            if processor_task.is_done() is False:
                self.logger.debug("Calling {0} callbacks".format(len(self.callback_list)))
                for callback in self.callback_list:
                    callback(self)
            else:
                self.stop_processing()

    def stop_processing(self):
        self.stop_processing_event.set()

    def set_roi(self, roi_cent, roi_dim):
        self.roi_cent = roi_cent
        self.roi_dim = roi_dim

    def set_processing_parameters(self, threshold, cal, kernel):
        self.threshold = threshold
        self.kernel = kernel
        self.cal = cal


ScanParam = namedtuple("ScanParam", "scan_attr_name scan_device_name scan_start_pos scan_end_pos scan_step "
                                    "scan_pos_tol scan_pos_check_interval "
                                    "measure_attr_name_list measure_device_list measure_number measure_interval")
ScanResult = namedtuple("ScanResult", "pos_list measure_list timestamp_list")


class ScanTask(Task):
    def __init__(self, scan_param, device_handler, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        # type: (ScanParam) -> None
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.scan_param = scan_param
        self.device_handler = device_handler
        self.scan_result = None

    def action(self):
        self.logger.info("{0} starting scan of {1} from {2} to {3}. ".format(self, self.scan_param.scan_attr_name,
                                                                             self.scan_param.scan_start_pos,
                                                                             self.scan_param.scan_end_pos))
        self.logger.info("{0} measuring {1}. ".format(self, self.scan_param.measure_attr_name_list))
        pos_list = list()
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
                read_task = TangoReadAttributeTask(meas_attr, self.scan_param.measure_device_list[meas_ind],
                                                   self.device_handler, name="read_{0}".format(meas_attr),
                                                   timeout=self.timeout)
                rep_task = RepeatTask(read_task, self.scan_param.measure_number, self.scan_param.measure_interval,
                                      name="rep_{0}".format(meas_attr), timeout=self.timeout)
                measure_task_list.append(rep_task)
            measure_bag_task = BagOfTasksTask(measure_task_list, name="measure_bag", timeout=self.timeout)
            step_sequence_task = SequenceTask([write_pos_task, monitor_pos_task, measure_bag_task], name="step_seq")
            step_sequence_task.start()
            step_result = step_sequence_task.get_result(wait=True, timeout=self.timeout)
            if step_sequence_task.is_cancelled() is True:
                self.cancel()
                break
            if self.is_cancelled() is True:
                break
            pos_list.append(step_result[1].value)
            timestamp_list.append(step_result[1].time)
            meas_list.append(step_result[2])
            next_pos += self.scan_param.scan_step
            if next_pos > self.scan_param.scan_end_pos:
                self.event_done.set()

        self.scan_result = ScanResult(pos_list, meas_list, timestamp_list)
        self.result = self.scan_result


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
             "scan"]
    test = "scan"
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
        t1 = LoadImageTask(image_name, path_name, name="load_im")
        t1.start()

    elif test == "load_im_dir":
        image_name = "03_03_1.035_.png"
        path_name = "..\\..\\emittancesinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t1 = LoadQuadScanDirTask(path_name, "quad_dir")
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
        t1 = ScanTask(scan_param, handler, name="scan_task", timeout=3.0)
        t1.start()
