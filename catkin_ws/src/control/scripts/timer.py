#!/usr/bin/env python


# timer.py
import datetime as dt

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, duration=None):
        self._start_time = None
        self._duration = duration

    def start(self, duration):
        """Start a new timer"""
        self._duration = duration
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = dt.datetime.now()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self._start_time = None

    def reset(self):
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self.stop()
        self.start(self._duration)


    def is_timeout(self):
        """Checks whether the timer has been running longer than the allotted duration. """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        if self._duration is not None:
            return (dt.datetime.now() - self._start_time).total_seconds() > self._duration
        else:
            return False
