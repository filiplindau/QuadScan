"""
Created 2018-10-24

Tasks for async sequencing.

@author: Filip Lindau
"""

import threading
import uuid
import logging

logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class Task(object):
    def __init__(self, name=None, task_type="thread", trigger_list=[]):
        self.id = uuid.uuid1()
        self.name = name
        self.type = task_type
        self.trigger_list = trigger_list
        self.trigger_list.append(self.id)
        self.trigger_done_list = list()
        self.trigger_result_list = list()
        self.listener_list = list()
        self.lock = threading.Lock()
        self.started = False

    def add_trigger(self, trigger_id):
        logger.info("{0} {1} adding trigger {2}".format(type(self), self.id, trigger_id))
        with self.lock:
            self.trigger_list.append(trigger_id)

    def add_listener(self, listener):
        logger.info("{0} {1} adding listener {2}".format(type(self), self.id, listener.id))
        with self.lock:
            self.listener_list.append(listener)

    def remove_listener(self, listener):
        logger.info("{0} {1} removing listener {2}".format(type(self), self.id, listener))
        with self.lock:
            if type(listener) == uuid.UUID:
                for l in self.listener_list:
                    if l.id == listener:
                        self.listener_list.remove(l)
                        break
            else:
                try:
                    self.listener_list.remove(listener)
                except ValueError:
                    pass

    def action(self):
        pass

    def emit(self, result):
        logger.info("{0} {1} done. Emitting signal".format(type(self), self.id))
        for listener in self.listener_list:
            listener.update(self.id, result)

    def start(self):
        logger.info("{0} {1} starting.".format(type(self), self.id))
        with self.lock:
            self.started = True
        self.update(self.id)

    def update(self, task_id, result=None):
        logger.info("{0} {1} updating trigger_done list.".format(type(self), self.id))
        with self.lock:
            if task_id not in self.trigger_done_list:
                self.trigger_done_list.append(task_id)
                self.trigger_result_list.append(result)
        if len(self.trigger_list) == len(self.trigger_done_list):
            self.action()

    def cancel(self):
        logger.info("{0} {1} cancelling.".format(type(self), self.id))
        with self.lock:
            self.started = False
            self.trigger_done_list = list()
            self.trigger_result_list = list()

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
    def __init__(self, delay, name=None):
        Task.__init__(self, name)
        self.delay = delay

    def action(self):
        delayed_call = threading.Timer(self.delay, self.emit, [None])
        delayed_call.start()


if __name__ == "__main__":
    t = DelayTask(1.0)
    t.start()
