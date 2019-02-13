"""
Created 2018-12-06

Tasks for async sequencing.

@author: Filip Lindau
"""

import threading
import multiprocessing
try:
    import Queue
except ModuleNotFoundError:
    import queue as Queue
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


# From: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadWithExc(threading.Thread):
    """A thread class that supports raising exception in the thread from
       another thread.
    """
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL : this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do : self.ident

        raise AssertionError("could not determine the thread's id")

    def raise_exc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raise_exc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raise_exc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL : this function is executed in the context of the
        caller thread, to raise an exception in the context of the
        thread represented by this instance.
        """
        _async_raise(self._get_my_tid(), exctype)


def p_action(task_obj, queue):
    task_obj.action()
    queue.put(task_obj.get_result(wait=True))


class Task(object):
    class CancelException(Exception):
        pass

    def __init__(self, name=None, action_exec_type="thread", timeout=None, trigger_dict=dict(), callback_list=list()):
        """

        :param name: Optional name to identify the task
        :param action_exec_type: "thread" or "process" Determines if action method is executed in thread or process
        :param timeout: timeout in seconds that is spent waiting for triggers
        :param trigger_dict: Dict of tasks that need to trigger before executing the action method of this task
        :param callback_list:
        """
        self.id = uuid.uuid1()
        if name is None:
            name = str(self.id)
        self.name = name
        self.action_exec_type = action_exec_type
        self.trigger_dict = dict()
        self.trigger_done_list = list()
        self.trigger_result_dict = dict()
        self.lock = threading.Lock()
        self.event_done = threading.Event()
        self.trigger_event = threading.Event()
        self.started = False
        self.completed = False
        self.cancelled = False
        self.result = None
        self.timeout = timeout
        self.run_thread = None
        self.callback_list = callback_list

        self.logger = logging.getLogger("Task.{0}".format(self.name.upper()))
        self.logger.setLevel(logging.DEBUG)

        for e in trigger_dict.values():
            self.add_trigger(e)

    def add_trigger(self, trigger_task):
        self.logger.info("{0} adding trigger {1}".format(self, trigger_task.name))
        with self.lock:
            self.trigger_dict[trigger_task.id] = trigger_task

    def add_callback(self, callback):
        self.logger.info("{0} adding callback {1}".format(self, callback))
        with self.lock:
            if callback not in self.callback_list:
                self.callback_list.append(callback)
        # Callback immediately if we are already finished
        if self.is_done() is True:
            callback(self)

    def remove_callback(self, callback):
        self.logger.info("{0} removing callback {1}".format(self, callback))
        with self.lock:
            if callback in self.callback_list:
                self.callback_list.remove(callback)

    def clear_callback_list(self):
        with self.lock:
            self.callback_list = list()

    def get_done_event(self):
        return self.event_done

    def get_signal_event(self):
        return self.signal_event

    def get_result(self, wait, timeout=-1):
        if self.completed is not True:
            if wait is True:
                if timeout > 0:
                    self.event_done.wait(timeout)
                else:
                    self.event_done.wait()
        return self.result

    def action(self):
        self.logger.info("{0} entering action.".format(self))
        self.result = None

    def emit(self):
        self.logger.debug("{0} done. Emitting signal".format(self))

        self.event_done.set()
        self.logger.debug("{0}: Calling {1} callbacks".format(self, len(self.callback_list)))
        for callback in self.callback_list:
            callback(self)

    def run(self):
        self.logger.debug("{0} entering run.".format(self))

        # Setup threads for triggers:
        already_done_flag = True
        for tr_id in self.trigger_dict:
            already_done_flag = False
            th = threading.Thread(target=self._wait_trigger, args=(self.trigger_dict[tr_id],))
            th.start()
        if already_done_flag is False:
            # Wait for trigger_event:
            if self.timeout is not None:
                completed_flag = self.trigger_event.wait(self.timeout)
                if completed_flag is False:
                    self.logger.info("{0} {1} timed out.".format(type(self), self.name))
                    self.cancel()
                    return
            else:
                self.trigger_event.wait()

        if self.is_cancelled() is True:
            return

            self.logger.debug("{0} triggers ready".format(self))
        try:
            if self.action_exec_type == "thread":
                self.action()
                self.logger.debug("{0} action done".format(self))
            else:
                queue = multiprocessing.Queue(2)
                process = multiprocessing.Process(target=p_action, args=(self, queue))
                process.start()
                self.result = queue.get(block=True, timeout=self.timeout)
                self.logger.debug("{0} action process done".format(self))
        except self.CancelException:
            self.logger.info("{0} Cancelled".format(self))
            return
        except Exception as e:
            self.logger.error("{0} exception: {1}".format(self, traceback.format_exc()))
            self.result = e
            self.result = traceback.format_exc()
            self.cancel()
            return
        with self.lock:
            self.completed = True
            self.started = False
        self.emit()

    def start(self):
        self.logger.debug("{0} starting.".format(self))
        with self.lock:
            self.trigger_done_list = list()
            self.trigger_result_dict = dict()
            self.event_done.clear()
            self.trigger_event.clear()
            self.started = True
            self.completed = False
            self.cancelled = False
            self.result = None
        # self.run_thread = threading.Thread(target=self.run)
        self.run_thread = ThreadWithExc(target=self.run)
        self.run_thread.start()

    def cancel(self):
        self.logger.debug("{0} cancelling.".format(self))
        with self.lock:
            self.started = False
            self.completed = False
            self.cancelled = True
            self.trigger_done_list = list()
            self.trigger_result_dict = dict()
            self.trigger_event.set()
            self.run_thread.raise_exc(self.CancelException)
        self.emit()

    def is_cancelled(self):
        return self.cancelled

    def is_done(self):
        return self.completed

    def is_started(self):
        return self.started

    def _wait_trigger(self, trigger_task):
        self.logger.debug("{0} starting wait for task {1}.".format(self, trigger_task.name))
        e = trigger_task.get_done_event()
        e.wait()
        # Cancel task if trigger was cancelled:
        if trigger_task.is_cancelled() is True:
            self.cancel()
            return
        with self.lock:
            self.trigger_done_list.append(trigger_task.id)
            if len(self.trigger_done_list) == len(self.trigger_dict):
                self.trigger_event.set()

    def __eq__(self, other):
        """
        Check if this task is the same as another based on the id, name or object
        :param other: Other task id number, name, or task object
        :return: True if it is the same
        """
        if type(other) == uuid.UUID:
            eq = self.id == other
        elif type(other) == str:
            eq = self.name == other
        elif type(other) == Task:
            eq = self.id == other.id
        else:
            eq = False
        return eq

    def __repr__(self):
        s = "{0} {1}".format(type(self), self.name)
        return s

    def __str__(self):
        s = "{0} {1}".format(type(self).__name__, self.name)
        return s


class SubclassTask(Task):
    """
    Copy for subclassing convenience
    """

    def __init__(self, name=None, timeout=None, trigger_dict=dict(),
                 callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)

    def action(self):
        self.logger.info("{0} entering action.".format(self))
        self.result = None


class DelayTask(Task):
    def __init__(self, delay, name=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, trigger_dict=trigger_dict, callback_list=callback_list)
        self.delay = delay

    def action(self):
        self.logger.info("{0} Starting delay of {1} s".format(self, self.delay))
        time.sleep(self.delay)
        self.result = True


class CallableTask(Task):
    def __init__(self, call_func, call_args=list(), call_kwargs=dict(),
                 name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.callable = call_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs

        self.logger.setLevel(logging.WARN)

    def action(self):
        self.logger.info("{0} calling {1}. ".format(self, self.callable))
        # Exceptions are caught in the parent run thread.
        # Still, I want to log an error message and re-raise
        try:
            res = self.callable(*self.call_args, **self.call_kwargs)
        except Exception as e:
            self.logger.error("Error when executing {0} with args {1}, {2}:\n {3}".format(self.callable,
                                                                                          self.call_args,
                                                                                          self.call_kwargs,
                                                                                          e))
            raise e
        self.result = res


class RepeatTask(Task):
    """
    Repeat a task a number of times and store the last result.
    If intermediate results are needed: wait for event_done emission
    from the repeating task. Use repetitions <= 0 for infinite number of
    repeats.
    """
    def __init__(self, task, repetitions, delay=0, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.task = task                    # type: Task
        self.repetitions = repetitions
        self.delay = delay

    def action(self):
        self.logger.info("{0} repeating task {1} {2} times.".format(self, self.task, self.repetitions))
        current_rep = 0
        result_list = list()
        while current_rep < self.repetitions:
            self.task.start()
            res = self.task.get_result(wait=True, timeout=self.timeout)
            # Check if cancelled or error..
            if self.task.is_cancelled() is True:
                self.cancel()
                break
            if self.is_cancelled() is True:
                break
            if self.repetitions >= 0:
                # Only store intermediate results in a list if there is not an infinite number of repetitions.
                result_list.append(res)
            time.sleep(self.delay)
            current_rep += 1
        self.result = result_list

    def cancel(self):
        if self.task.is_cancelled() is False:
            self.task.cancel()
        Task.cancel(self)


class SequenceTask(Task):
    """
    Run a sequence of tasks in a list and emit done event after all
    are completed.

    The results of the tasks are stored in a list.
    """
    def __init__(self, task_list, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.task_list = task_list
        self.logger.setLevel(logging.INFO)

    def action(self):
        self.logger.info("{0} running task sequence of length {1}.".format(self, len(self.task_list)))
        res = list()

        for t in self.task_list:
            t.start()
            res.append(t.get_result(wait=True))
            # Check if cancelled or error..
            if t.is_cancelled() is True:
                self.cancel()
                break
            if self.is_cancelled() is True:
                break

        self.result = res


class BagOfTasksTask(Task):
    """
    Run a list of tasks and emit done event after all
    are completed.

    The results of the tasks are stored in a list.
    """
    def __init__(self, task_list, name=None, timeout=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.task_list = task_list

    def action(self):
        self.logger.info("{0} running {1} tasks.".format(self, len(self.task_list)))
        res = list()

        for t in self.task_list:
            t.start()
        for t in self.task_list:
            # Task results are stored in a list in the order they were created.
            res.append(t.get_result(wait=True))
            # Check if cancelled or error..
            if t.is_cancelled() is True:
                self.cancel()
                break
            if self.is_cancelled() is True:
                break

        self.result = res


class MapToProcessTask(Task):
    """
    Map a number of tasks to pool of processes.

    The results of the tasks are stored in a list.
    """

    def __init__(self, f, data_list, number_processes=5, name=None, trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, trigger_dict=trigger_dict, callback_list=callback_list)
        self.f = f
        self.data_list = data_list
        self.lock = multiprocessing.Lock()
        self.num_proc = number_processes
        self.pool

    def action(self):
        self.logger.info("{0} executing function on data of length {1} across {2} processes.".format(self,
                                                                                                     len(self.task_list),
                                                                                                     self.num_proc))
        res = list()
        pool = multiprocessing.Pool(processes=self.num_proc)
        with self.lock:
            res = pool.map(self.f, self.data_list)

        self.result = res

    def add_data(self, data_list):
        with self.lock:
            self.data_list.append(data_list)

    def set_data_list(self, data_list):
        with self.lock:
            self.data_list = data_list


def clearable_pool_worker(in_queue, out_queue):
    while True:
        task = in_queue.get()
        f = task[0]
        args = task[1]
        kwargs = task[2]
        proc_id = task[3]
        # logger.debug("Pool worker executing {0} with args {1}, {2}".format(f, args, kwargs))
        # logger.debug("Pool worker executing {0} ".format(f))
        try:
            retval = f(*args, **kwargs)
            # logger.debug("Putting {0} on out_queue".format(retval))
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

        self.logger.setLevel(logging.INFO)

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
        while self.completed_work_items < self.next_process_id:
            self.logger.debug("{0}: Waiting for {1} work items".format(self, self.next_process_id - self.completed_work_items))
            time.sleep(0.01)
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
            retval = result[0]
            proc_id = result[1]
            self.completed_work_items += 1
            try:
                with self.lock:
                    self.result_dict[proc_id] = retval
            except KeyError as e:
                self.logger.error("Work item id {0} not found.".format(proc_id))
                raise e
            if isinstance(retval, Exception):
                raise retval
            # self.logger.debug("{0}: result_dict {1}".format(self, pprint.pformat(self.result_dict)))
            self.result = retval
            for callback in self.callback_list:
                callback(self)


class ThreadPoolTask(Task):
    """
    Start up a thread pool that can consume data. Emit trigger when queue is empty.

    The results of the tasks are stored in a list.
    """

    def __init__(self, work_func, number_threads=8, name=None, timeout=None,
                 trigger_dict=dict(), callback_list=list()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict, callback_list=callback_list)
        self.work_func = work_func
        self.lock = threading.Lock()
        self.finish_threads_event = threading.Event()
        self.pending_work_items = list()

        self.in_queue = Queue.Queue()
        self.out_queue = Queue.Queue()
        self.num_threads = number_threads
        self.threads = None
        self.next_thread_id = 0
        self.completed_work_items = 0
        self.id_lock = threading.Lock()

        self.result_thread = None
        self.stop_result_thread_flag = False
        self.result_dict = dict()

        self.logger.setLevel(logging.INFO)

    def run(self):
        self.create_threads()
        Task.run(self)

    def start(self):
        self.next_thread_id = 0
        Task.start(self)

    def action(self):
        self.logger.info("{0}: starting thread pool for "
                         "executing function {1} across {2} threads.".format(self, self.work_func, self.num_threads))
        self.logger.info("{1}: Starting {0} threads".format(self.num_threads, self))
        self.completed_work_items = 0
        for p in self.threads:
            p.start()
        self.stop_result_thread_flag = False
        self.result_thread = threading.Thread(target=self.result_thread_func)
        self.result_thread.start()

        self.finish_threads_event.wait(self.timeout)
        if self.finish_threads_event.is_set() is False:
            self.cancel()
        while self.completed_work_items < self.next_thread_id:
            time.sleep(0.01)
        self.stop_threads()
        self.result = self.result_dict

    def add_work_item(self, *args, **kwargs):
        self.logger.info("{0}: Adding work item".format(self))
        # self.logger.debug("{0}: Args: {1}, kwArgs: {2}".format(self, args, kwargs))
        thread_id = self.next_thread_id
        self.next_thread_id += 1
        self.in_queue.put((self.work_func, args, kwargs, thread_id))
        self.result_dict[thread_id] = None
        self.logger.debug("{0}: Queue added. Thread id: {1}".format(self, thread_id))

    def create_threads(self):
        if self.threads is not None:
            self.stop_threads()

        self.logger.info("{1}: Creating {0} threads".format(self.num_threads, self))
        p_list = list()
        for p in range(self.num_threads):
            p = threading.Thread(target=self.clearable_pool_worker, args=(self.in_queue, self.out_queue))
            p_list.append(p)
        self.threads = p_list

    def stop_threads(self):
        self.logger.info("{0}: Stopping threads".format(self))
        if self.threads is not None:
            for p in self.threads:
                p.join(1.0)
        # self.processes = None
        self.stop_result_thread_flag = True
        try:
            self.result_thread.join(1.0)
        except AttributeError:
            pass
        self.result_thread = None

    def finish_processing(self):
        self.finish_threads_event.set()

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
            except Queue.Empty:
                self.logger.debug("In-queue empty")
                break

    def result_thread_func(self):
        self.logger.debug("{0}: Starting result collection thread".format(self))
        while self.stop_result_thread_flag is False:
            try:
                result = self.out_queue.get(True, 0.1)
            except Queue.Empty:
                continue
            retval = result[0]
            proc_id = result[1]
            self.completed_work_items += 1
            try:
                self.result_dict[proc_id] = retval
            except KeyError as e:
                self.logger.error("Work item id {0} not found.".format(proc_id))
                raise e
            if isinstance(retval, Exception):
                raise retval
            self.logger.debug("{0}: result_dict {1}".format(self, pprint.pformat(self.result_dict)))
            self.result = retval
            for callback in self.callback_list:
                callback(self)

    def clearable_pool_worker(self, in_queue, out_queue):
        while self.finish_threads_event.is_set() is False:
            task = in_queue.get()
            f = task[0]
            args = task[1]
            kwargs = task[2]
            proc_id = task[3]
            # logger.debug("Pool worker executing {0} with args {1}, {2}".format(f, args, kwargs))
            # logger.debug("Pool worker executing {0} ".format(f))
            try:
                retval = f(*args, **kwargs)
                # logger.debug("Putting {0} on out_queue".format(retval))
                # logger.debug("Putting {0} result on out_queue".format(f))
            except Exception as e:
                retval = e
                logger.error("{0} Error {1} ".format(f, retval))
            out_queue.put((retval, proc_id))
