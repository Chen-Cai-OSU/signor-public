# Created at 2020-03-28
# Summary: set tensorboard dir
import os

from signor.ioio.dir import tb_dir


class tb_util():
    def __init__(self):
        self.dir = tb_dir()

    def fname(self):
        """ return the fname for writer """

        return os.path.join(self.dir, 'runs', '')

if __name__ == '__main__':
    print(tb_util().fname())