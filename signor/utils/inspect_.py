import inspect
from pprint import pprint

__all__ = ['class_name', 'func_name', 'method_name', 'class_name_of_method']


def class_name(instance_or_class):
    if isinstance(instance_or_class, type):
        return func_name(instance_or_class)
    return func_name(instance_or_class.__class__)


def func_name(func):
    try:
        return func.__module__ + '.' + func.__qualname__
    except:
        return str(func)



def method_name(method):
    assert '.' in method.__qualname__, '"{}" is not a method.'.format(repr(method))
    return func_name(method)


def class_name_of_method(method):
    name = method_name(method)
    return name[:name.rfind('.')]


class test():
    def __init__(self):
        pass

    def length(self, x):
        return len(x)

    def attr(self):
        return 10

class MyClass(object):
    a = '12'
    b = '34'

    def myfunc(self):
        return self.a



def classlookup(cls):
    # https://stackoverflow.com/questions/1401661/list-all-base-classes-in-a-hierarchy-of-given-class
    # todo: refactor
    try:
         c = list(cls.__bases__)
         if len(c) ==1 and c[0] == object:
             print(c[0])
         elif len(c) ==1 and c[0]!=object:
             print(c[0])
         elif len(c) > 1:
             print(c)
         else:
             pass

         for base in c:
             classlookup(base)
             # c.extend(classlookup(base))
    except AttributeError: # no __bases__
        print('AttributeError')



if __name__ == '__main__':
    c = 10
    classlookup(int)

    exit()
    x = MyClass()
    x.c=10
    attrs = dir(x)
    filter_attrs = [attr for attr in attrs if not (attr.startswith('__') and attr.endswith('__'))]
    print(filter_attrs)
    for attr in filter_attrs:
        tmp = getattr(x, attr)
        print(attr, tmp, type(tmp))
    exit()

    # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    attributes = inspect.getmembers(MyClass, lambda a: not (inspect.isroutine(a)))
    pprint(attributes)

    print([a[0] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])

    exit()
    func = len
    print(func_name(func))

    t = test
    print(func_name(t.attr))
    print(func_name(t.length))