""" keep files fresh """
import os

def remove_file(f):
    if os.path.exists(f):
        print(f"removed existing file {f}!")
        os.remove(f)
    else:
        print(f'{f} does not exist. Do nothing.')


