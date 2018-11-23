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


# From: https://stackoverflow.com/questions/12317940/python-threading-can-i-sleep-on-two-threading-events-simultaneously

def or_set(self):
    self._set()
    self.changed()


def or_clear(self):
    self._clear()
    self.changed()


def orify(e, changed_callback):
    try:
        callback_list = e.callback_list
    except AttributeError:
        callback_list = list()
        e._set = e.set
        e._clear = e.clear
    callback_list.append(changed_callback)

    def changed():
        [callback() for callback in callback_list]

    e.changed = changed
    e.set = lambda: or_set(e)
    e.clear = lambda: or_clear(e)


def make_or_event(or_event, *events):

    # or_event = threading.Event()
    def changed():
        bools = [e.is_set() for e in events]
        if any(bools):
            or_event.set()
        else:
            or_event.clear()
    for e in events:
        orify(e, changed)
    changed()
    return or_event


class Task(object):
    def __init__(self, name=None, task_type="thread", timeout=None, trigger_dict=dict()):
        self.id = uuid.uuid1()
        if name is None:
            name = self.id
        self.name = name
        self.type = task_type
        self.trigger_dict = dict()
        self.trigger_done_list = list()
        self.trigger_result_list = list()
        self.listener_list = list()
        self.lock = threading.Lock()
        self.event_done = threading.Event()
        self.cancel_event = threading.Event()
        self.trigger_event = threading.Event()
        make_or_event(self.trigger_event, self.cancel_event)
        self.started = False
        self.completed = False
        self.cancelled = False
        self.result = None
        self.timeout = timeout
        for e in trigger_dict.itervalues():
            self.add_trigger(e)

    def add_trigger(self, trigger_task):
        logger.info("{0} {1} adding trigger {2}".format(type(self), self.name, trigger_task.id))
        e = trigger_task.get_event()
        with self.lock:
            self.trigger_dict[trigger_task.id] = e
            make_or_event(self.trigger_event, e)

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
        while len(self.trigger_done_list) != len(self.trigger_dict):
            if self.timeout is not None:
                completed_flag = self.trigger_event.wait(self.timeout)
                if completed_flag is False:

                    self.cancel()
                    return
            else:
                self.trigger_event.wait()

            for tr_id in self.trigger_dict:
                if tr_id not in self.trigger_done_list:
                    task = self.trigger_dict[tr_id]
                    if task.is_cancelled() is True:
                        self.cancel()
                        return
                    self.trigger_done_list.append(tr_id)
        self.action()
        self.emit()

    def start(self):
        logger.info("{0} {1} starting.".format(type(self), self.name))
        with self.lock:
            self.started = True
            self.completed = False
            self.cancelled = False
        t = threading.Thread(target=self.run)
        t.start()

    def update(self, task_id, result=None):
        logger.info("{0} {1} updating trigger_done list.".format(type(self), self.name))

        with self.lock:
            if task_id not in self.trigger_done_list:
                self.trigger_done_list.append(task_id)
                self.trigger_result_list.append(result)
        if len(self.trigger_dict) == len(self.trigger_done_list):
            self.action()

    def cancel(self):
        logger.info("{0} {1} cancelling.".format(type(self), self.name))
        with self.lock:
            self.started = False
            self.completed = False
            self.cancelled = True
            self.trigger_done_list = list()
            self.trigger_result_list = list()
            self.cancel_event.set()
        self.emit()

    def is_cancelled(self):
        return self.cancelled

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


if __name__ == "__main__":
    t1 = DelayTask(1.0, name="task1")
    t2 = DelayTask(2.0, name="task2", trigger_dict={"delay": t1})
    t2.start()
    t1.start()
