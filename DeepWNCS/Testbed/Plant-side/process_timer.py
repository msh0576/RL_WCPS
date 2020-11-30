# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:52:17 2020

@author: Sihoon
"""

from multiprocessing import Process, Event

# based on the threading timer class by Itamar Shtull-Trauring
class Timer(Process):
    """Calls a function after a specified number of seconds:

        t = Timer(30.0, f, args=None, kwargs=None)
        t.start()
        t.cancel() #stops the timer if it is still waiting

    """
    def __init__(self, interval, function, args=None, kwargs=None, iterations=1, infinite=False):
        Process.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = Event()
        self.infinite = infinite

        if infinite:
            self.iterations = infinite
            self.current_iteration = infinite
        else:
            self.iterations = iterations
            self.current_iteration = 1

    def cancel(self):
        """Stop the timer if it hasn't already finished."""
        self.finished.set()

    def run(self):
        while not self.finished.is_set() and self.current_iteration <= self.iterations:
            self.finished.wait(self.interval)
            if not self.finished.is_set():
                self.function(*self.args, **self.kwargs)
            if not self.infinite:
                self.current_iteration += 1
        self.finished.set()