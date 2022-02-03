import argparse
from time import sleep

parser = argparse.ArgumentParser(description='test script')
parser.add_argument('--t', type=float, default=1, help='sleep time')

if __name__ == '__main__':
    args = parser.parse_args()
    sleep(args.t)
