import six
import collections

def merge_iterable(v1, v2):
    assert issubclass(type(v1), type(v2)) or issubclass(type(v2), type(v1))
    if isinstance(v1, (dict, set)):
        v = v1.copy().update(v2)
        return v

    return v1 + v2


def dict_deep_update(a, b):
    for key in b:
        if key in a and type(b[key]) is dict:
            dict_deep_update(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def assert_none(ins, msg=None):
    msg = msg or '{} is not None'.format(ins)
    assert ins is None, msg


def assert_notnone(ins, msg=None, name='instance'):
    msg = msg or '{} is None'.format(name)
    assert ins is not None, msg


def prop2func(method_name):
    return lambda x: getattr(x, method_name)()


class test():
    def __init__(self):
        pass

    def length(self, x):
        return len(x)

    def attr(self):
        return 10


def map_exec(func, *iterables):
    """ like apply in r """
    return list(map(func, *iterables))


def filter_exec(func, iterable):
    return list(filter(func, iterable))


def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)

def filterdict():
    # todo: implement
    pass


if __name__ == '__main__':
    import numpy as np
    from signor.utils.random_ import fix_seed
    fix_seed()
    print(np.random.random())
    exit()

    t = test()
    print(t.length([1, 23]))
    attr = prop2func('attr')
    print(attr(t))
    exit()

    print(merge_iterable([1, 2], [3, 6]))
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 3, 'b': 2}
    print(dict_deep_update(d1, d2))
