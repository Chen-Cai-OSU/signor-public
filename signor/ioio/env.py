""" virtual environment related """
import os
conda = '/home/cai.507/anaconda3/'

def env_python(env=None):
    """ return the python interpreter of environment """

    if env is not None:
        return os.path.join(conda, 'envs', env, 'bin', 'python')
    else:
        return os.path.join(conda, 'bin', 'python')

if __name__ == '__main__':
    print(env_python(env='pretrain'))