"""
Created 2018-10-24

Tasks for async sequencing.

@author: Filip Lindau
"""

import threading
import uuid
import logging
import time
import ctypes
import inspect
try:
    import PyTango as pt
except ImportError:
    import tango as pt

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


class Task(object):
    class CancelException(Exception):
        pass

    def __init__(self, name=None, task_type="thread", timeout=None, trigger_dict=dict()):
        self.id = uuid.uuid1()
        if name is None:
            name = self.id
        self.name = name
        self.task_type = task_type
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
        for e in trigger_dict.values():
            self.add_trigger(e)

    def add_trigger(self, trigger_task):
        logger.info("{0} adding trigger {1}".format(self, trigger_task.name))
        with self.lock:
            self.trigger_dict[trigger_task.id] = trigger_task

    def get_event(self):
        return self.event_done

    def get_result(self, wait=False, timeout=-1):
        if self.completed is not True:
            if wait is True:
                if timeout > 0:
                    self.event_done.wait(timeout)
                else:
                    self.event_done.wait()
        return self.result

    def action(self):
        logger.info("{0} entering action.".format(self))
        self.result = None

    def emit(self):
        logger.info("{0} done. Emitting signal".format(self))
        with self.lock:
            self.completed = True
            self.event_done.set()

    def run(self):
        logger.info("{0} entering run.".format(self))

        # Setup threads for triggers:
        already_done_flag = True
        for tr_id in self.trigger_dict:
            already_done_flag = False
            th = threading.Thread(target=self._wait_trigger, args=(self.trigger_dict[tr_id],))
            th.start()
        if already_done_flag is False:
            if self.timeout is not None:
                completed_flag = self.trigger_event.wait(self.timeout)
                if completed_flag is False:
                    logger.info("{0} {1} timed out.".format(type(self), self.name))
                    self.cancel()
                    return
            else:
                self.trigger_event.wait()

        if self.is_cancelled() is True:
            return

        logger.debug("triggers ready")
        try:
            self.action()
            logger.debug("action ready")
        except self.CancelException:
            logger.info("{0} Cancelled".format(self))
            return
        except Exception as e:
            logger.error("{0} exception: {1}".format(self, e))
            self.result = e
            self.cancel()
            return
        logger.debug("emit now")
        self.emit()
        logger.debug("emitted")

    def start(self):
        logger.debug("{0} starting.".format(self))
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
        logger.debug("{0} cancelling.".format(self))
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

    def _wait_trigger(self, trigger_task):
        logger.debug("{0} starting wait for task {1}.".format(self, trigger_task.name))
        e = trigger_task.get_event()
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


class DelayTask(Task):
    def __init__(self, delay, name=None, trigger_dict=dict()):
        Task.__init__(self, name, trigger_dict=trigger_dict)
        self.delay = delay

    def action(self):
        logger.info("{0} Starting delay of {1} s".format(self, self.delay))
        time.sleep(self.delay)
        self.result = True


class CallableTask(Task):
    def __init__(self, call_func, call_args=list(), call_kwargs=dict(),
                 name=None, trigger_dict=dict()):
        Task.__init__(self, name, trigger_dict=trigger_dict)
        self.callable = call_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs

    def action(self):
        logger.info("{0} calling {1}. ".format(self, self.callable))
        # Exceptions are caught in the parent run thread.
        # Still, I want to log an error message and re-raise
        try:
            res = self.callable(*self.call_args, **self.call_kwargs)
        except Exception as e:
            logger.error("Error when executing {0} with args {1}, {2}:\n {3}".format(self.callable,
                                                                                     self.call_args,
                                                                                     self.call_kwargs,
                                                                                     e))
            raise e
        self.result = res


class RepeatTask(Task):
    """
    Repeat a task a number of times and store the last result.
    If intermediate results are needed: wait for event_done emission
    from the repeating task.
    """
    def __init__(self, task, repetitions, name=None, trigger_dict=dict()):
        Task.__init__(self, name, trigger_dict=trigger_dict)
        self.task = task
        self.repetitions = repetitions

    def action(self):
        logger.info("{0} repeating task {1} {2} times.".format(self, self.task, self.repetitions))
        for i in range(self.repetitions):
            self.task.start()
            self.task.get_event().wait()
            # Check if cancelled or error..
        self.result = self.task.get_result()


class SequenceTask(Task):
    """
    Run a sequence of tasks in a list and emit done event after all
    are completed.

    The results of the tasks are stored in a list.
    """
    def __init__(self, task_list, name=None, trigger_dict=dict()):
        Task.__init__(self, name, trigger_dict=trigger_dict)
        self.task_list = task_list

    def action(self):
        logger.info("{0} running task sequence of length {1}.".format(self, len(self.task_list)))
        res = list()
        for t in self.task_list:
            t.start()
            res.append(t.get_result(True))
            # Check if cancelled or error..

        self.result = res


class TangoDeviceConnectTask(Task):
    def __init__(self, device_name, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.device_name = device_name

    def action(self):
        logger.info("{0} entering action. ".format(self))
        # Exceptions are caught in the parent run thread.
        logger.debug("Connecting to {0}".format(self.device_name))
        dev = pt.DeviceProxy(self.device_name)
        self.result = dev


class TangoReadAttributeTask(Task):
    def __init__(self, attribute_name, device_name, device_handler, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler

    def action(self):
        logger.info("{0} reading {1} on {2}. ".format(self, self.attribute_name, self.device_name))
        dev = self.device_handler.get_device(self.device_name)
        attr = dev.read_attribute(self.attribute_name)
        self.result = attr


class TangoWriteAttributeTask(Task):
    def __init__(self, attribute_name, value, device_name, device_handler, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.device_name = device_name
        self.attribute_name = attribute_name
        self.device_handler = device_handler
        self.value = value

    def action(self):
        logger.info("{0} writing {1} to {2} on {2}. ".format(self,
                                                             self.value,
                                                             self.attribute_name,
                                                             self.device_name))
        dev = self.device_handler.get_device(self.device_name)
        res = dev.write_attribute(self.attribute_name, self.value)
        self.result = res


class TangoMonitorAttributeTask(Task):
    def __init__(self, device, attribute, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.device = device
        self.attribute = attribute

    def action(self):
        logger.info("{0} entering action. ".format(self))
        self.result = True


class LoadImageTask(Task):
    def __init__(self, image_name, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.image_name = image_name

    def action(self):
        logger.info("{0} entering action. ".format(self))
        self.result = True


class ProcessImageTask(Task):
    def __init__(self, image, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.image = image

    def action(self):
        logger.info("{0} entering action. ".format(self))
        self.result = True


class ScanTask(Task):
    def __init__(self, scan_params, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, trigger_dict=trigger_dict)
        self.scan_params = scan_params

    def action(self):
        logger.info("{0} entering action. ".format(self))
        self.result = True


class DeviceHandler(object):
    """
    Handler for open devices.
    Devices are stored in a dict for easy retrieval.
    New devices are added asynchronously with add_device method.
    """
    def __init__(self, tango_host=None, name=None):
        self.devices = dict()
        self.tango_host = tango_host
        if name is None:
            self.name = self.__repr__()
        else:
            self.name = name

    def get_device(self, device_name):
        logger.debug("{0} Returning device {1}".format(self, device_name))
        try:
            dev = self.devices[device_name]
        except KeyError:
            # Maybe this should just raise an exception instead of auto-adding:
            dev = self.add_device(device_name)
        return dev

    def add_device(self, device_name):
        """
        Add a device to the open devices dictionary.
        A device connect task is created and started.

        :param device_name: Tango name of device
        :return: opened device proxy
        """
        logger.info("{0} Adding device {1} to device handler".format(self, device_name))
        if device_name in self.devices:
            logger.debug("Device already in dict. No need")
            return True
        if self.tango_host is not None:
            full_dev_name = "{0}/{1}".format(self.tango_host, device_name)
        else:
            full_dev_name = device_name
        task = TangoDeviceConnectTask(full_dev_name, name=device_name)
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
        dev = task.get_result()
        logger.info("{0} {1} Device connection completed. Returned {1}".format(self, device_name, dev))
        self.devices[device_name] = dev

    def __str__(self):
        s = "{0} {1}".format(type(self).__name__, self.name)
        return s


if __name__ == "__main__":
    tests = ["delay", "dev_handler", "exc"]
    test = "dev_handler"
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

        t3 = TangoWriteAttributeTask("double_scalar_w", 10.0, dev_name, handler, trigger_dict={"1": t2})
        t4 = TangoReadAttributeTask("double_scalar_w", dev_name, handler, trigger_dict={"1": t3})
        t4.start()
        t3.start()
        logger.info("Double scalar: {0}".format(t4.get_result()))
        logger.info("Double scalar: {0}".format(t4.get_result(True).value))

        t6 = SequenceTask([t2, DelayTask(2.0, "delay_seq")])
        t5 = RepeatTask(t6, 10)
        t5.start()
        time.sleep(5)
        t6.cancel()

    elif test == "exc":
        t1 = DelayTask(2.0, name="long delay")
        t1.start()
        time.sleep(1.0)
        t1.cancel()
