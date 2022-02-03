# Created at 2020-04-12
# Summary: timeout mechanism. https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish

import errno
import os
import signal
import signal
import time
from datetime import timedelta
from functools import wraps
from timeit import default_timer

import numpy as np

from signor.monitor.time import timefunc


class Timer(object):
    """Timer class.

    `Original code <https://github.com/miguelgfierro/pybase/blob/2298172a13fb4a243754acbc6029a4a2dcf72c20/log_base/timer.py>`_.
    Examples:
        >>> import time
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> with Timer() as t:
        ...   time.sleep(1)
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 1...'
    """

    def __init__(self, name=None, v=True, ):
        self._timer = default_timer
        self._interval = 0
        self.running = False
        self.name = name
        self.v = v

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = self._timer()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )
        if self.v and self.name is not None:
            print("Took {:0.5f} seconds for {}.".format(self.interval, self.name))

    @property
    def interval(self):
        """Get time interval in seconds.

        Returns:
            float: Seconds.
        """
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            return self._interval


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TimeoutError(Exception):
    pass

def timeout2(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            print(f'{func.__name__} takes more than {seconds}s. Exit.\n')
            # exit()
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timefunc
@timeout2(seconds=4)
def longfunc(t):
    time.sleep(t)
    print(f'Finish sleep {t}s.')

if __name__ == '__main__':
    with Timer('test'):
        time.sleep(1.4)
    exit()
    for t in range(10):
        longfunc(t)

    exit()
    for _ in range(10):
        with timeout(seconds=3, error_message='Timeout (3) for random sleep.'):
            t = 4 * np.random.random()
            time.sleep(t)
            print(f'Done sleep {t}')
