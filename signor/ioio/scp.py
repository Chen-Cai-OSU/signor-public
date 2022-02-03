' scp ./DoorDash_challenge.ipynb chen@169.228.63.12:/home/chen/fromosu/Signor/data/notebooks/hanfu'
from signor.utils.cli import runcmd

source = 'chen@169.228.63.12:/home/chen/fromosu/Signor/signor/graph/IGN/invariantgraphnetworks-pytorch/main_scripts/fig'
dest = '/Users/admin/Documents/osu/Research/IGN-Convergence/paper/IGN-Convergence/icml-version/'
cmd = f'scp -r {source} {dest}'
runcmd(cmd)
