# -*- coding:utf-8 -*-
"""
Created on Mar 19, 2018

@author: Filip Lindau
"""
import threading
import time
import logging
import traceback
import multiprocessing

from twisted_cut.protocol import Protocol, Factory
from twisted_cut import reflect, defer, error
try:
    import PyTango.futures as tangof
except ImportError:
    import tango.futures as tangof

logger = logging.getLogger("TangoTwisted")
logger.setLevel(logging.DEBUG)
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def deferred_from_future(future):
    d = defer.Deferred()

    def callback(future):
        e = future.exception()
        if e:
            d.errback(e)
            return

        d.callback(future.result())

    future.add_done_callback(callback)
    return d


class TangoAttributeProtocol(Protocol):
    def __init__(self, operation, name, data=None, **kw):
        self.data = data
        self.kw = kw
        self.result_data = None
        self.operation = operation
        self.name = name
        self.d = None
        self.factory = None

        self._check_target_value = None
        self._check_period = 0.3
        self._check_timeout = 1.0
        self._check_tolerance = None
        self._check_starttime = None
        self._check_lastreadtime = None
        self._check_timeout_deferred = None

        self.logger = logging.getLogger("SpectrometerCameraController.Protocol_{0}_{1}".format(operation.upper(), name))
        self.logger.setLevel(logging.WARNING)

    def makeConnection(self, transport=None):
        self.logger.debug("Protocol {0} make connection".format(self.name))
        if self.operation == "read":
            self.d = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
        elif self.operation == "write":
            self.d = deferred_from_future(self.factory.device.write_attribute(self.name, self.data, wait=False))
        elif self.operation == "command":
            self.d = deferred_from_future(self.factory.device.command_inout(self.name, self.data, wait=False))
        elif self.operation == "check":
            self.d = self.check_attribute()
        self.d.addCallbacks(self.dataReceived, self.connectionLost)
        return self.d

    def dataReceived(self, data):
        self.result_data = data
        self.logger.debug("Received data {0}".format(data))
        return data

    def connectionLost(self, reason):
        self.logger.debug("Connection lost, reason {0}".format(reason))
        return reason

    # def check_attribute(self, attr_name, dev_name, target_value, period=0.3, timeout=1.0, tolerance=None, write=True):
    def check_attribute(self):
        """
        Check an attribute to see if it reaches a target value. Returns a deferred for the result of the
        check.
        Upon calling the function the target is written to the attribute if the "write" parameter is True.
        Then reading the attribute is polled with the period "period" for a maximum time of "timeout".
        If the read value is within tolerance, the callback deferred is fired.
        If the read value is outside tolerance after "timeout" time, the errback is fired.
        The maximum time to check is "timeout"

        :param attr_name: Tango name of the attribute to check, e.g. "position"
        :param dev_name: Tango device name to use, e.g. "gunlaser/motors/zaber01"
        :param target_value: Attribute value to wait for
        :param period: Polling period when checking the value
        :param timeout: Time to wait for the attribute to reach target value
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :return: Deferred that will fire depending on the result of the check
        """
        self.logger.info("Entering check_attribute")
        self.d = defer.Deferred()

        try:
            write = self.kw["write"]
        except KeyError:
            write = True
        try:
            self._check_timeout = self.kw["timeout"]
        except KeyError:
            self.logger.debug("No timeout specified, using 1.0 s")
            self._check_timeout = 1.0
        try:
            self._check_tolerance = self.kw["tolerance"]
        except KeyError:
            self._check_tolerance = None
            self.logger.debug("No tolerance specified, using None")
        try:
            self._check_period = self.kw["period"]
        except KeyError:
            self._check_period = 0.3
            self.logger.debug("No period specified, using 0.3 s")
        try:
            self._check_target_value = self.kw["target_value"]
        except KeyError:
            self._check_target_value = None
            self.logger.error("No target value specified")
            self.d.errback("Target value required")
            return self.d

        self._check_starttime = time.time()
        self._check_lastreadtime = self._check_starttime
        self._check_timeout_deferred = defer_later(self._check_timeout, self.d.errback,
                                                   RuntimeError("Check {0} timeout".format(self.name)))

        if write is True:
            self.logger.debug("Issuing initial write")
            dw = deferred_from_future(self.factory.device.write_attribute(self.name,
                                                                          self._check_target_value,
                                                                          wait=False))
            # Add a callback that starts the reading after write completes
            dw.addCallbacks(self._check_do_read, self._check_fail)
        else:
            self.logger.debug("Issuing initial read")
            dw = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
            dw.addCallbacks(self._check_read_done, self._check_fail)

        # Return the deferred that will fire with the result of the check
        return self.d

    def _check_w_done(self, result):
        self.logger.info("Write done, result {0}".format(result))
        return result

    def _check_fail(self, err):
        self.logger.error("Error, result {0}".format(err))
        self.d.errback(err)
        return err

    def _check_do_read(self, result=None):
        self.logger.info("Issuing read ")
        dr = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
        dr.addCallbacks(self._check_read_done, self._check_fail)

    def _check_read_done(self, result):
        self.logger.info("Read done, result {0}".format(result))
        t0 = time.time()
        # First check if we timed out. Then fire the errback function and exit.
        if t0 - self._check_starttime > self._check_timeout:
            self.logger.warning("Timeout exceeded")
            self.d.errback("timeout")
            return
        # Now try extracting the read value (sometimes this is None).
        try:
            val = result.value
        except AttributeError:
            self.d.errback("Read result error {0}".format(result))
            return

        # Check if the read value is within tolerance. Then fire the callback and exit.
        done = False
        if self._check_tolerance is None:
            if val == self._check_target_value:
                done = True
        else:
            if abs(val - self._check_target_value) < self._check_tolerance:
                done = True
        if done is True:
            self.logger.debug("Result {0} with tolerance of target value {1}".format(val, self._check_target_value))
            try:
                self._check_timeout_deferred.cancel()
            except AttributeError as e:
                self.logger.warning("AttributeError when cancelling timeout deferred: {0}".format(e))
            self.d.callback(result)
            return

        # Finally calculate the wait time until next read.
        last_duration = t0 - self._check_lastreadtime
        if last_duration > self._check_period:
            delay = 0
        else:
            delay = self._check_period - last_duration
        self.logger.debug("Delay until next read: {0}".format(delay))
        defer_later(delay, self._check_do_read)
        self._check_lastreadtime = t0


class TangoAttributeFactory(Factory):
    protocol = TangoAttributeProtocol

    def __init__(self, device_name):
        self.device_name = device_name
        self.device = None
        self.connected = False
        self.d = None
        self.attribute_dict = dict()

        self.logger = logging.getLogger("SpectrometerCameraController.Factory_{0}".format(device_name))
        self.logger.setLevel(logging.WARNING)

    def startFactory(self):
        self.logger.info("Starting TangoAttributeFactory")
        self.d = deferred_from_future(tangof.DeviceProxy(self.device_name, wait=False))
        self.d.addCallbacks(self.connection_success, self.connection_fail)
        return self.d

    def buildProtocol(self, operation, name, data=None, d=None, **kw):
        """
        Create a TangoAttributeProtocol that sends a Tango operation to the factory deviceproxy.

        :param operation: Tango attribute operation, e.g. read, write, command
        :param name: Name of Tango attribute
        :param data: Data to send to Tango device, if any
        :param d: Optional deferred to add the result of the Tango operation to
        :return: Deferred that fires when the Tango operation is completed.
        """
        if self.connected is True:
            self.logger.info("Connected, create protocol and makeConnection")
            self.logger.debug("args: {0}, {1}, {2}, kw: {3}".format(operation, name, data, kw))
            proto = self.protocol(operation, name, data, **kw)
            proto.factory = self
            df = proto.makeConnection()
            df.addCallbacks(self.data_received, self.protocol_fail)
            if d is not None:
                df.addCallback(d)
        else:
            self.logger.debug("Not connected yet, adding to connect callback")
            # df = defer.Deferred()
            self.d.addCallbacks(self.build_protocol_cb, self.connection_fail, callbackArgs=[operation, name, data],
                                callbackKeywords=kw)
            df = self.d

        return df

    def build_protocol_cb(self, result, operation, name, data, df=None, **kw):
        """
        We need this extra callback for buildProtocol since the first argument
        is always the callback result.

        :param result: Result from deferred callback. Ignore.
        :param operation: Tango attribute operation, e.g. read, write, command
        :param name: Name of Tango attribute
        :param data: Data to send to Tango device, if any
        :param df: Optional deferred to add the result of the Tango operation to
        :return: Deferred that fires when the Tango operation is completed.
        """
        self.logger.debug("Now call build protocol")
        d = self.buildProtocol(operation, name, data, df, **kw)
        return d

    def connection_success(self, result):
        self.logger.debug("Connected to deviceproxy")
        self.connected = True
        self.device = result

    def connection_fail(self, err):
        self.logger.error("Failed to connect to device. {0}".format(err))
        self.device = None
        self.connected = False
        return err

    def protocol_fail(self, err):
        self.logger.error("Failed to do attribute operation on device {0}: {1}".format(self.device_name, err))
        # fail = Failure(err)
        return err

    def data_received(self, result):
        self.logger.debug("Data received: {0}".format(result))
        try:
            self.attribute_dict[result.name] = result
        except AttributeError:
            pass
        self.logger.debug("Exiting data_received.")
        return result

    def get_attribute(self, name):
        if name in self.attribute_dict:
            self.logger.debug("Attribute {0} already read. Retrieve from dictionary.".format(name))
            d = defer.Deferred()
            d.callback(self.attribute_dict[name])
        else:
            self.logger.debug("Attribute {0} not in dictionary, retrieve it from device".format(name))
            d = self.buildProtocol("read", name)
        return d


class LoopingCall(object):
    def __init__(self, loop_callable, *args, **kw):
        self.f = loop_callable
        self.args = args
        self.kw = kw
        self.running = False
        self.interval = 0.0
        self.starttime = None
        self._deferred = None
        self._runAtStart = False
        self.call = None

        self.loop_deferred = defer.Deferred()

        self.logger = logging.getLogger("TangoTwisted.LoopingCall")
        self.logger.setLevel(logging.WARNING)

    def start(self, interval, now=True):
        """
        Start running function every interval seconds.
        @param interval: The number of seconds between calls.  May be
        less than one.  Precision will depend on the underlying
        platform, the available hardware, and the load on the system.
        @param now: If True, run this call right now.  Otherwise, wait
        until the interval has elapsed before beginning.
        @return: A Deferred whose callback will be invoked with
        C{self} when C{self.stop} is called, or whose errback will be
        invoked when the function raises an exception or returned a
        deferred that has its errback invoked.
        """
        if self.running is True:
            self.stop()

        self.logger.debug("Starting looping call")
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.running = True
        # Loop might fail to start and then self._deferred will be cleared.
        # This why the local C{deferred} variable is used.
        deferred = self._deferred = defer.Deferred()
        self.starttime = time.time()
        self.interval = interval
        self._runAtStart = now
        if now:
            self()
        else:
            self._schedule_from(self.starttime)
        return deferred

    def stop(self):
        """Stop running function.
        """
        assert self.running, ("Tried to stop a LoopingCall that was "
                              "not running.")
        self.running = False
        if self.call is not None:
            self.call.cancel()
            self.call = None
            d, self._deferred = self._deferred, None
            d.callback(self)

    def reset(self):
        """
        Skip the next iteration and reset the timer.
        @since: 11.1
        """
        assert self.running, ("Tried to reset a LoopingCall that was "
                              "not running.")
        if self.call is not None:
            self.call.cancel()
            self.call = None
            self.starttime = time.time()
            self._schedule_from(self.starttime)

    def __call__(self):

        def cb(result):
            if self.running:
                self._schedule_from(time.time())
                new_loop_deferred = defer.Deferred()
                for callback in self.loop_deferred.callbacks:
                    new_loop_deferred.callbacks.append(callback)
                self.loop_deferred.callback(result)
                self.loop_deferred = new_loop_deferred
            else:
                df, self._deferred = self._deferred, None
                df.callback(self)

        def eb(failure):
            self.running = False
            df, self._deferred = self._deferred, None
            self.logger.error("Looping call error: {0}".format(failure))

            df.errback(failure)

        self.call = None
        self.logger.debug("Looping call")
        d = defer.maybeDeferred(self.f, *self.args, **self.kw)
        d.addCallback(cb)
        d.addErrback(eb)

    def _schedule_from(self, when):
        """
        Schedule the next iteration of this looping call.
        @param when: The present time from whence the call is scheduled.
        """

        def how_long():
            # How long should it take until the next invocation of our
            # callable?  Split out into a function because there are multiple
            # places we want to 'return' out of this.
            if self.interval == 0:
                # If the interval is 0, just go as fast as possible, always
                # return zero, call ourselves ASAP.
                return 0
            # Compute the time until the next interval; how long has this call
            # been running for?
            running_for = when - self.starttime
            # And based on that start time, when does the current interval end?
            until_next_interval = self.interval - (running_for % self.interval)
            # Now that we know how long it would be, we have to tell if the
            # number is effectively zero.  However, we can't just test against
            # zero.  If a number with a small exponent is added to a number
            # with a large exponent, it may be so small that the digits just
            # fall off the end, which means that adding the increment makes no
            # difference; it's time to tick over into the next interval.
            if when == when + until_next_interval:
                # If it's effectively zero, then we need to add another
                # interval.
                return self.interval
            # Finally, if everything else is normal, we just return the
            # computed delay.
            return until_next_interval
        self.call = threading.Timer(how_long(), self)
        self.call.start()


class DeferredCondition(object):
    def __init__(self, condition, cond_callable, *args, **kw):
        if "result" in condition:
            self.condition = condition
        else:
            self.condition = "result " + condition
        self.cond_callable = cond_callable
        self.args = args
        self.kw = kw
        self.logger = logging.getLogger("SpectrometerCameraController.DeferredCondition")
        self.logger.setLevel(logging.WARNING)

        self.running = False
        self.call_timer = None
        self._deferred = None
        self.starttime = None
        self.interval = None
        self.clock = None

    def start(self, interval, timeout=None):
        if self.running is True:
            self.stop()

        self.logger.debug("Starting checking condition {0}".format(self.condition))
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.running = True
        # Loop might fail to start and then self._deferred will be cleared.
        # This why the local C{deferred} variable is used.
        deferred = self._deferred = defer.Deferred()
        if timeout is not None:
            self.clock = ClockReactorless()
            deferred.addTimeout(timeout, self.clock)
            deferred.addErrback(self.cond_error)
        self.starttime = time.time()
        self.interval = interval
        self._run_callable()
        return deferred

    def stop(self):
        """Stop running function.
        """
        assert self.running, ("Tried to stop a LoopingCall that was "
                              "not running.")
        self.running = False
        if self.call_timer is not None:
            self.call_timer.cancel()
            self.call_timer = None
            d, self._deferred = self._deferred, None
            d.callback(None)

    def _run_callable(self):
        self.logger.debug("Calling {0}".format(self.cond_callable))
        d = defer.maybeDeferred(self.cond_callable, *self.args, **self.kw)
        d.addCallbacks(self.check_condition, self.cond_error)

    def _schedule_from(self, when):
        """
        Schedule the next iteration of this looping call.
        @param when: The present time from whence the call is scheduled.
        """
        t = 0
        if self.interval > 0:
            # Compute the time until the next interval; how long has this call
            # been running for?
            running_for = when - self.starttime
            # And based on that start time, when does the current interval end?
            until_next_interval = self.interval - (running_for % self.interval)
            # Now that we know how long it would be, we have to tell if the
            # number is effectively zero.  However, we can't just test against
            # zero.  If a number with a small exponent is added to a number
            # with a large exponent, it may be so small that the digits just
            # fall off the end, which means that adding the increment makes no
            # difference; it's time to tick over into the next interval.
            if when == when + until_next_interval:
                # If it's effectively zero, then we need to add another
                # interval.
                t = self.interval
            # Finally, if everything else is normal, we just return the
            # computed delay.
            else:
                t = until_next_interval
        self.logger.debug("Scheduling new function call in {0} s".format(t))
        self.call = threading.Timer(t, self._run_callable)
        self.call.start()

    def check_condition(self, result):
        self.logger.debug("Checking condition {0} with result {1}".format(self.condition, result))
        if self.running is True:
            cond = eval(self.condition)
            self.logger.debug("Condition evaluated {0}".format(cond))
            if cond is True:
                d, self._deferred = self._deferred, None
                d.callback(result)
            else:
                self._schedule_from(time.time())
            return result
        else:
            return False

    def cond_error(self, err):
        self.logger.error("Condition function returned error {0}".format(err))
        self.running = False
        if self.call_timer is not None:
            self.call_timer.cancel()
            self.call_timer = None
        d, self._deferred = self._deferred, None
        d.errback(err)
        return err


def defer_later(delay, delayed_callable, *a, **kw):
    # logger.info("Calling {0} in {1} seconds".format(delayed_callable, delay))

    def defer_later_cancel(deferred):
        delayed_call.cancel()

    d = defer.Deferred(defer_later_cancel)
    d.addCallback(lambda ignored: delayed_callable(*a, **kw))
    delayed_call = threading.Timer(delay, d.callback, [None])
    delayed_call.start()
    return d


class DelayedCallReactorless(object):
    # enable .debug to record creator call stack, and it will be logged if
    # an exception occurs while the function is being run
    debug = False
    _str = None

    def __init__(self, time, func, args, kw={}, cancel=None, reset=None,
                 seconds=time.time):
        """
        @param time: Seconds from the epoch at which to call C{func}.
        @param func: The callable to call.
        @param args: The positional arguments to pass to the callable.
        @param kw: The keyword arguments to pass to the callable.
        @param cancel: A callable which will be called with this
            DelayedCall before cancellation.
        @param reset: A callable which will be called with this
            DelayedCall after changing this DelayedCall's scheduled
            execution time. The callable should adjust any necessary
            scheduling details to ensure this DelayedCall is invoked
            at the new appropriate time.
        @param seconds: If provided, a no-argument callable which will be
            used to determine the current time any time that information is
            needed.
        """
        self.time, self.func, self.args, self.kw = time, func, args, kw
        self.resetter = reset
        self.canceller = cancel
        self.seconds = seconds
        self.cancelled = self.called = 0
        self.delayed_time = 0
        self.timer = None
        if self.debug:
            self.creator = traceback.format_stack()[:-2]
        self._schedule_call()

    def getTime(self):
        """Return the time at which this call will fire
        @rtype: C{float}
        @return: The number of seconds after the epoch at which this call is
        scheduled to be made.
        """
        return self.time + self.delayed_time

    def _schedule_call(self):
        if self.timer is not None:
            if self.timer.is_alive() is True:
                self.timer.cancel()
        self.timer = threading.Timer(self.time + self.delayed_time - time.time(), self._fire_call)
        self.timer.start()

    def _fire_call(self):
        self.called = True
        self.func(*self.args, **self.kw)

    def cancel(self):
        """Unschedule this call
        @raise AlreadyCancelled: Raised if this call has already been
        unscheduled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            if self.timer is not None:
                if self.timer.is_alive() is True:
                    self.timer.cancel()
            if self.canceller is not None:
                self.canceller(self)
            self.cancelled = 1
            if self.debug:
                self._str = str(self)
            del self.func, self.args, self.kw

    def reset(self, secondsFromNow):
        """Reschedule this call for a different time
        @type secondsFromNow: C{float}
        @param secondsFromNow: The number of seconds from the time of the
        C{reset} call at which this call will be scheduled.
        @raise AlreadyCancelled: Raised if this call has been cancelled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            new_time = self.seconds() + secondsFromNow
            if new_time < self.time:
                self.delayed_time = 0
                self.time = new_time
                if self.resetter is not None:
                    self.resetter(self)
            else:
                self.delayed_time = new_time - self.time
            self._schedule_call()

    def delay(self, secondsLater):
        """Reschedule this call for a later time
        @type secondsLater: C{float}
        @param secondsLater: The number of seconds after the originally
        scheduled time for which to reschedule this call.
        @raise AlreadyCancelled: Raised if this call has been cancelled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            self.delayed_time += secondsLater
            if self.delayed_time < 0:
                self.activate_delay()
                self.resetter(self)

    def activate_delay(self):
        self.time += self.delayed_time
        self.delayed_time = 0
        self._schedule_call()

    def active(self):
        """Determine whether this call is still pending
        @rtype: C{bool}
        @return: True if this call has not yet been made or cancelled,
        False otherwise.
        """
        return not (self.cancelled or self.called)

    def __le__(self, other):
        """
        Implement C{<=} operator between two L{DelayedCall} instances.
        Comparison is based on the C{time} attribute (unadjusted by the
        delayed time).
        """
        return self.time <= other.time

    def __lt__(self, other):
        """
        Implement C{<} operator between two L{DelayedCall} instances.
        Comparison is based on the C{time} attribute (unadjusted by the
        delayed time).
        """
        return self.time < other.time

    def __str__(self):
        if self._str is not None:
            return self._str
        if hasattr(self, 'func'):
            # This code should be replaced by a utility function in reflect;
            # see ticket #6066:
            if hasattr(self.func, '__qualname__'):
                func = self.func.__qualname__
            elif hasattr(self.func, '__name__'):
                func = self.func.func_name
                if hasattr(self.func, 'im_class'):
                    func = self.func.im_class.__name__ + '.' + func
            else:
                func = reflect.safe_repr(self.func)
        else:
            func = None

        now = self.seconds()
        L = ["<DelayedCall 0x%x [%ss] called=%s cancelled=%s" % (
                id(self), self.time - now, self.called,
                self.cancelled)]
        if func is not None:
            L.extend((" ", func, "("))
            if self.args:
                L.append(", ".join([reflect.safe_repr(e) for e in self.args]))
                if self.kw:
                    L.append(", ")
            if self.kw:
                L.append(", ".join(['%s=%s' % (k, reflect.safe_repr(v)) for (k, v) in self.kw.items()]))
            L.append(")")

        if self.debug:
            L.append("\n\ntraceback at creation: \n\n%s" % ('    '.join(self.creator)))
        L.append('>')

        return "".join(L)


class ClockReactorless(object):
    """
    Provide a deterministic, easily-controlled implementation of
    L{IReactorTime.callLater}.  This is commonly useful for writing
    deterministic unit tests for code which schedules events using this API.
    """

    rightNow = 0.0

    def __init__(self):
        self.calls = []
        self.timer = None

    def seconds(self):
        """
        Pretend to be time.time().  This is used internally when an operation
        such as L{IDelayedCall.reset} needs to determine a time value
        relative to the current time.
        @rtype: C{float}
        @return: The time which should be considered the current time.
        """
        self.rightNow = time.time()
        return self.rightNow

    def _sortCalls(self):
        """
        Sort the pending calls according to the time they are scheduled.
        """
        self.calls.sort(key=lambda a: a.getTime())

    def callLater(self, when, what, *a, **kw):
        """
        See L{twisted.internet.interfaces.IReactorTime.callLater}.
        """

        # def defer_later_cancel(deferred):
        #     delayed_call.cancel()
        #
        # dc = defer.Deferred(defer_later_cancel)
        # dc.addCallback(lambda ignored: what(*a, **kw))
        # delayed_call = threading.Timer(when, dc.callback, [None])
        # delayed_call.start()
        # self.calls.append(dc)
        # self._sortCalls()

        dc = DelayedCallReactorless(self.seconds() + when,
                                    what, a, kw,
                                    self.calls.remove,
                                    lambda c: None,
                                    self.seconds)
        self.calls.append(dc)
        self._sortCalls()
        return dc

    def getDelayedCalls(self):
        """
        See L{twisted.internet.interfaces.IReactorTime.getDelayedCalls}
        """
        return self.calls

    def advance(self, amount):
        """
        Move time on this clock forward by the given amount and run whatever
        pending calls should be run.
        @type amount: C{float}
        @param amount: The number of seconds which to advance this clock's
        time.
        """
        self.rightNow += amount
        # self._sortCalls()
        while self.calls and self.calls[0].getTime() <= self.seconds():
            call = self.calls.pop(0)
            call.called = 1
            call.func(*call.args, **call.kw)
            # self._sortCalls()

    def pump(self, timings):
        """
        Advance incrementally by the given set of times.
        @type timings: iterable of C{float}
        """
        for amount in timings:
            self.advance(amount)


def defer_to_thread(f, *args, **kwargs):
    """
    Run a function in a thread and return the result as a Deferred.
    @param f: The function to call.
    @param *args: positional arguments to pass to f.
    @param **kwargs: keyword arguments to pass to f.
    @return: A Deferred which fires a callback with the result of f,
    or an errback with a L{twisted.python.failure.Failure} if f throws
    an exception.
    """
    def run_thread(df, func, *f_args, **f_kwargs):
        try:
            logger.debug("Calling function {0} in thread.".format(func))
            result = func(*f_args, **f_kwargs)
            logger.debug("Thread deferred function returned {0}".format(result))
            df.callback(result)
        except Exception as e:
            df.errback(e)
    logger.info("Deferring function {0} to thread.".format(f))
    d = defer.Deferred()
    rt_args = (d, f) + args
    t = threading.Thread(target=run_thread, args=rt_args, kwargs=kwargs)
    t.start()
    return d


def f_wrapper(f, *f_args, **f_kwargs):
    try:
        result = f(*f_args, **f_kwargs)
    except Exception as e:
        result = e
    return result


def defer_to_pool(pool, f, *args, **kwargs):
    df = defer.Deferred()

    def pool_callback(result):
        if isinstance(result, Exception):
            df.errback(result)
        else:
            df.callback(result)

    logger.info("Deferring function {0} to process pool.".format(f))
    args_wrapper = (f,) + args
    pool.apply_async(f_wrapper, args=args_wrapper, kwds=kwargs, callback=pool_callback)
    # pool.apply_async(f, args=args, kwds=kwargs, callback=pool_callback)
    return df


def clearable_pool_worker(in_queue, out_queue):
    while True:
        task = in_queue.get()
        f = task[0]
        args = task[1]
        kwargs = task[2]
        d = task[3]
        try:
            retval = f(*args, **kwargs)
        except Exception as e:
            retval = e
        out_queue.put((retval, d))


class ClearablePool(object):
    def __init__(self, processes=multiprocessing.cpu_count()):
        self.logger = logging.getLogger("TangoTwisted.ClearablePool")
        self.logger.setLevel(logging.WARNING)
        self.logger.info("ClearablePool.__init__")

        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.num_processes = processes
        self.processes = None
        self.next_process_id = 0
        self.id_lock = threading.Lock()
        self.deferred_dict = dict()

        self.result_thread = None
        self.stop_thread_flag = False

        self.start_processes()

    def start_processes(self):
        if self.processes is not None:
            self.stop_processes()

        self.stop_thread_flag = False
        self.result_thread = threading.Thread(target=self.result_thread_func)
        self.result_thread.start()

        self.logger.info("Starting {0} processes".format(self.num_processes))
        p_list = list()
        for p in range(self.num_processes):
            p = multiprocessing.Process(target=clearable_pool_worker, args=(self.in_queue, self.out_queue))
            p.start()
            p_list.append(p)
        self.processes = p_list

    def stop_processes(self):
        self.logger.info("Stopping processes")
        if self.processes is not None:
            for p in self.processes:
                p.terminate()
        # self.processes = None
        self.stop_thread_flag = True
        try:
            self.result_thread.join(1.0)
        except AttributeError:
            pass
        self.result_thread = None

    def stop(self):
        self.stop_processes()

    def close(self):
        self.stop_processes()

    def add_task(self, f, *args, **kwargs):
        d = defer.Deferred()
        self.logger.debug("Putting task in queue. Args: {0}".format(args))
        with self.id_lock:
            id = self.next_process_id
            self.next_process_id += 1
        self.in_queue.put((f, args, kwargs, id))
        self.deferred_dict[id] = d
        return d

    def clear_pending_tasks(self):
        self.logger.info("Clearing pending tasks")
        while self.in_queue.empty() is False:
            print("Not empty")
            try:
                self.logger.debug("get no wait")
                task = self.in_queue.get_nowait()
                id = task[3]
                self.logger.debug("Removing task {0}".format(id))
                try:
                    self.deferred_dict.pop(id)
                except KeyError:
                    pass
            except multiprocessing.queues.Empty:
                self.logger.debug("In-queue empty")
                break

    def result_thread_func(self):
        while self.stop_thread_flag is False:
            try:
                result = self.out_queue.get(True, 0.1)
            except multiprocessing.queues.Empty:
                continue
            self.logger.debug("Result: {0}".format(result))
            retval = result[0]
            id = result[1]
            try:
                d = self.deferred_dict.pop(id)
            except KeyError as e:
                self.logger.error("Task id {0} not found.".format(id))
                raise e
            if isinstance(retval, Exception):
                d.errback(retval)
            else:
                d.callback(retval)


def test_cb2(result):
    logger.info("Test CB2 result: {0}".format(result))
    return result


t0 = time.time()
t1 = time.time()
count = 0


def looping_test(a):
    t = time.time()
    global t0
    global count
    count += 1
    dt = t - t0
    if dt > 1.0:
        logger.info("Looping test count: {0}, dt: {1}".format(count, dt))
        t0 = t
    return count


def looping_test_cb(result):
    logger.info("Callback result: {0}".format(result))
    return result

    # t = time.time()
    # global t1
    # dt = t - t1
    # if dt > 1.0:
    #     logger.info("Callback result: {0}, dt: {1}".format(result, dt))
    #     t1 = t


def looping_test_eb(err):
    logger.error("Callback error: {0}".format(err))


def pool_test(a, b):
    time.sleep(0.5)
    return a/b


if __name__ == "__main__":
    t0 = time.time()
    count = 0
    # lc = LoopingCall(looping_test, count)
    # d = lc.start(0.001)
    dl = list()
    a = [1.0 * x + 1 for x in range(10)]
    b = [1.0 * x for x in range(10)]

    # import multiprocessing
    # pool = multiprocessing.Pool(processes=4)
    # for k in range(len(a)):
    #     d = defer_to_pool(pool, pool_test, a[k], b[k])
    #     d.addCallback(test_cb2)
    #     d.addErrback(looping_test_eb)
    #     dl.append(d)
    # pool.close()

    cp = ClearablePool(4)

    dl = list()
    for k in range(len(a)):
        print("a: {0}, b: {1}".format(a[k], b[k]))
        d = cp.add_task(pool_test, a[k], b[k])
        d.addCallback(test_cb2)
        d.addErrback(looping_test_eb)
    time.sleep(1.1)
    cp.clear_pending_tasks()

    # lc.loop_deferred.addCallback(looping_test_cb)
    # lc.loop_deferred.addErrback(looping_test_eb)
