import argparse
import sys
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--sample_size', type=int, default=10, help='sample size')
parser.add_argument('--chem_data', type=str, default='bace', help='chem dataset')

if __name__ == '__main__':
    sys.argv += ['']
    # print(sys.argv)
    args = parser.parse_args()
    print(args)