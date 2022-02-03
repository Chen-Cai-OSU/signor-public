""" launch notebook """
import argparse
import os

parser = argparse.ArgumentParser(description='launch jupyter notebook local or global')
parser.add_argument('--local', action='store_true', help='Disable CUDA')
parser.add_argument('--print', action='store_true', help='print only')


if __name__ == '__main__':
    args = parser.parse_args()

    dir_remote = '/home/cai.507/Documents/DeepLearning/Signor/'
    dir_local = '/Users/admin/Documents/osu/Research/Signor/'

    nb_file = 'signor/viz/ani/information_bottleneck.ipynb &' # change

    jupyter_remote = '~/anaconda3/envs/signor/bin/jupyter notebook --no-browser --port 1111 '
    jupyter_local = ' ~/anaconda3/bin/jupyter notebook '

    nb_remote, nb_local = dir_remote + nb_file, dir_local + nb_file
    nb = nb_local if args.local else nb_remote
    jupyter = jupyter_local if args.local else jupyter_remote

    cmd = ' '.join([jupyter, nb])
    print('executing following cmd...')
    print(cmd)

    if args.print:
        exit()
        os.system(cmd)
