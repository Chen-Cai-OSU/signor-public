""" generate a README when write to a file """
import os

from signor.format.format import timestamp
from signor.ioio.dir import mktemp
from signor.utils.inspect_ import func_name


def test_func():
    return 1


def readme(func, f, verbose=False, *args, **kwargs):
    """
    auto generate a README to f when calling function.
    :param func:
    :param f:
    :param args:
    :param kwargs:
    :return:
    """
    document = f"This is a auto-generated README when calling function {func_name(func)} " \
        f"'with args {args} and kwargs {kwargs}\n at {timestamp()}"
    with open(f, "w") as text_file:
        text_file.write(document)

    if verbose:
        print(f'write to file {f} with success')
    print(document)


if __name__ == '__main__':
    f = mktemp(verbose=True, tmp_dir=True)
    f = os.path.join(f, 'tmp')
    print(f)
    readme(test_func, f)
