import datetime
import os
import platform
import signal
import time

import pandas as pd

from signor.format.format import pf, format_counter


def timefunc(method, threshold=1, precision=2, scale=1):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if int(te - ts) > threshold:
                print(f'{method.__name__}: {pf((te - ts) * scale, precision=precision)}s')
        return result

    return timed


class timefunc_global(object):
    def __init__(self, threshold=-1, precision=2, scale=1, topk=2):
        from collections import Counter
        self.d = Counter()  # time
        self.cnt = Counter()  # how many times
        self.threshold = threshold
        self.precision = precision
        self.scale = scale
        self.topk = topk

    def tf(self, method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if int(te - ts) > self.threshold:
                func, t = f'{method.__qualname__}', pf((te - ts) * self.scale, precision=self.precision)
                print(f'{func}: {t}s')
                self.d[func] += t
                self.cnt[func] += 1

                topk_cnt = self.d.most_common(self.topk)
                print(f'top {self.topk} bottleneck:')
                format_counter(topk_cnt, precision=3)

                print()
                topk_cnt = self.cnt.most_common(self.topk)
                print(f'top {self.topk} bottleneck (times):')
                format_counter(topk_cnt, precision=3)

                print('-' * 5)
            return result

        return timed

    def register(self, t, method):
        self.d[method] += t

    def dummy_tf(self):
        return lambda x: x


def complexity(method, n=[1, 10, 100]):
    def timed(*args, **kw):
        for _n in n:
            ts = time()
            print(args, kw)
            kw['n'] = _n
            print(kw)
            result = method(*args, **kw)
            te = time()
            print(f'n: {_n}. t: {pf(te - ts, 2)}')
        return result

    return timed


tf = lambda x: x


# tf = partial(timefunc, threshold=-1, precision=5)


def modification_date(filename):
    # http://bit.ly/380fxXI
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


@timefunc
def h():
    time.sleep(1.1)


def signal_handler(signum, frame):
    raise Exception("Timed out!")


tf_class = timefunc_global(topk=10, precision=5)
tf_global = tf_class.tf  # dummy_tf()


# tf_global = tf_class.dummy_tf()
# tf = tf_global

class TimeTester(object):
    def __init__(self):
        pass

    @tf_global
    def long_function_call(self, t=1.5):
        time.sleep(t)

    @tf_global
    def func(self, ):
        time.sleep(1.2)


from memory_profiler import profile


@profile
@tf_global
def func2():
    time.sleep(1.3)


def curtime():
    # current time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return st


def curtime_():
    # current time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M')
    return st


def today():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    return st


def yesterday():
    # https://bit.ly/3rmUCZc
    from datetime import datetime, timedelta
    yesterday = datetime.now() - timedelta(1)
    st = datetime.strftime(yesterday, '%Y-%m-%d')
    return st


def days_before(n=1):
    # https://bit.ly/3rmUCZc
    from datetime import datetime, timedelta
    yesterday = datetime.now() - timedelta(n)
    st = datetime.strftime(yesterday, '%Y-%m-%d')
    return st


def curdate():
    # current date
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    return st


def time_analysize(f='log'):
    with open(f, 'r') as f:
        cont = f.readlines()
    ret = []
    for l in cont:
        func, t = l.split(':')
        t = float(t.replace('s\n', ''))
        ret.append([func, t])
    df = pd.DataFrame(ret, columns=['func', 'time'])
    df['time'] = df['time'] / 60
    table = df.pivot_table(columns=['func'], values='time', aggfunc=[sum])
    print(table)
    return df


def toseconds(t):
    # t is something like '2009-09-26T01:51:42.000Z'
    # convert a timestamp to seconds
    # https://bit.ly/2XlpkI4
    ref = '2008-09-26T01:51:42.000Z'
    date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    t0 = datetime.datetime.strptime(ref, date_format)
    t1 = datetime.datetime.strptime(t, date_format)
    return (t1 - t0).total_seconds()


if __name__ == '__main__':
    print(toseconds('2009-09-26T01:51:42.000Z'))
    exit()
    t = TimeTester()
    t.long_function_call(t=2)
    t.func()
    t.long_function_call(t=1)
    t.long_function_call(t=0.1)
    t.func()
    tf_class.register(10, 'unknow method')
    func2()
    exit()
    print(today())
    print(yesterday())
    print(days_before(n=0))
    exit()
    date = curdate()
    print(date)
    exit()
    f = '/home/cai.507/Documents/DeepLearning/Signor/signor/monitor/log'
    time_analysize(f=f)

    exit()
    print(modification_date(__file__))
    print(creation_date(__file__))
    exit()
    print(curtime())
    exit()
    h()
    exit()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)  # Ten seconds
    long_function_call(t=3)
    try:
        long_function_call(t=3.5)
    except:
        print("Timed out!")
