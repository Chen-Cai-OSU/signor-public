from itertools import chain


def add_range(r1, r2):
    concatenated = chain(r1, r2)
    return concatenated


if __name__ == '__main__':
    r1, r2 = range(3), range(5)
    print(list(add_range(r1, r2)))
