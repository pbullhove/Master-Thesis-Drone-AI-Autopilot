#!/usr/bin/env python
import datetime as dt

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    """ A Timer class which is keeps track of whether a given duration has passed or not."""
    def __init__(self, duration=None):
        self._start_time = None
        self._duration = None
        if duration is not None:
            self.start(duration)

    def start(self, duration):
        """
        Start a new timer
            input: duration - number of seconds until timer is to timeOut
        """
        self._duration = duration
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = dt.datetime.now()

    def stop(self):
        """
        Stop the timer.
            input: None
            output: None
        Exception:
            Raises TimerError if timer is not running.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self._start_time = None

    def reset(self):
        """
        Stop the timer, and restart with the same duration.
            input: None
            output: None
        Exception:
            Raises TimerError if timer is not running.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self.stop()
        self.start(self._duration)


    def is_timeout(self):
        """
        Checks whether the timer has been running longer than the allotted duration.
            input: None
            output: Bool
        Exception:
            Raises TimerError if timer is not running.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        if self._duration is not None:
            return (dt.datetime.now() - self._start_time).total_seconds() > self._duration
        else:
            return False
