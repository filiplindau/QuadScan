"""
Created 2018-10-24

Tasks for async sequencing.

@author: Filip Lindau
"""

import threading
import uuid
import logging
import time

logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class Task(object):
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
        for e in trigger_dict.values():
            self.add_trigger(e)

    def add_trigger(self, trigger_task):
        logger.info("{0} {1} adding trigger {2}".format(type(self), self.name, trigger_task.name))
        with self.lock:
            self.trigger_dict[trigger_task.id] = trigger_task

    def get_event(self):
        return self.event_done

    def get_result(self):
        if self.completed is True:
            return self.result
        else:
            return None

    def action(self):
        logger.info("{0} {1} entering action.".format(type(self), self.name))
        self.result = None

    def emit(self):
        logger.info("{0} {1} done. Emitting signal".format(type(self), self.name))
        with self.lock:
            self.completed = True
            self.event_done.set()

    def run(self):
        logger.info("{0} {1} entering run.".format(type(self), self.name))

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

        try:
            self.action()
        except Exception as e:
            self.result = e
            self.cancel()
            return
        self.emit()

    def start(self):
        logger.debug("{0} {1} starting.".format(type(self), self.name))
        with self.lock:
            self.trigger_done_list = list()
            self.trigger_result_dict = dict()
            self.event_done.clear()
            self.trigger_event.clear()
            self.started = True
            self.completed = False
            self.cancelled = False
            self.result = None
        t = threading.Thread(target=self.run)
        t.start()

    def cancel(self):
        logger.debug("{0} {1} cancelling.".format(type(self), self.name))
        with self.lock:
            self.started = False
            self.completed = False
            self.cancelled = True
            self.trigger_done_list = list()
            self.trigger_result_dict = dict()
            self.trigger_event.set()
        self.emit()

    def is_cancelled(self):
        return self.cancelled

    def is_done(self):
        return self.completed

    def _wait_trigger(self, trigger_task):
        logger.debug("{0} {1} starting wait for task {2}.".format(type(self), self.name, trigger_task.name))
        e = trigger_task.get_event()
        e.wait()
        # if self.timeout is not None:
        #     completed_flag = e.wait(self.timeout)
        #     Cancel task if trigger timed out:
            # if completed_flag is False:
            #     self.cancel()
            #     return
        # else:
        #     e.wait()
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


class DelayTask(Task):
    def __init__(self, delay, name=None, trigger_dict=dict()):
        Task.__init__(self, name, trigger_dict=trigger_dict)
        self.delay = delay

    def action(self):
        logger.info("{0} {1} entering action. Starting delay of {2}s".format(type(self), self.name, self.delay))
        time.sleep(self.delay)
        self.result = True


class TangoDeviceConnectTask(Task):
    def __init__(self, device, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.device = device

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class TangoReadAttributeTask(Task):
    def __init__(self, device, attribute, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.device = device
        self.attribute = attribute

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class TangoWriteAttributeTask(Task):
    def __init__(self, device, attribute, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.device = device
        self.attribute = attribute

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class TangoMonitorAttributeTask(Task):
    def __init__(self, device, attribute, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.device = device
        self.attribute = attribute

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class ReadImageTask(Task):
    def __init__(self, image_name, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.image_name = image_name

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class ProcessImageTask(Task):
    def __init__(self, image, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.image = image

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


class ScanTask(Task):
    def __init__(self, scan_params, name=None, timeout=None, trigger_dict=dict()):
        Task.__init__(self, name, timeout=timeout, etrigger_dict=trigger_dict)
        self.scan_params = scan_params

    def action(self):
        logger.info("{0} {1} entering action. ".format(type(self), self.name))
        self.result = True


if __name__ == "__main__":
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

    # t2.cancel()
