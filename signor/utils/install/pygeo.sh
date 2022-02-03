#!/usr/bin/env bash
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

#TORCH='1.7.0' # '1.8.0' # '1.7.0'
#CUDA='cu110' # 'cu102' #'cu102' # 'cpu'

# learning_to_simulate
TORCH='1.8.0'
CUDA='cu102'

# test / hea-ml
TORCH='1.8.0'
CUDA='cu111' # 'cpu' #

TORCH='1.7.0'
CUDA='cu110' # 'cpu' #

CUDA='cu101'
TORCH='1.8.0'

CUDA='cu111'
TORCH='1.9.1'
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install --no-index --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

pip install --no-index --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-cache-dir torch-geometric
exit
#pip install --verbose --no-cache-dir torch-scatter
#pip install --verbose --no-cache-dir torch-sparse
#pip install --verbose --no-cache-dir torch-cluster
#pip install --verbose --no-cache-dir torch-spline-conv
#pip install torch-geometric

# requires pytorch=1.6.0
CUDA='cu110' #'cu102'
CUDATOOLKIT='11.0'
TORCH=1.7.1
flag='--ignore-installed --no-cache-dir'

#CUDA='cu101'
#TORCH=1.4.0
#pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-geometric
#conda install cudatoolkit=10.2

conda install cudatoolkit=$CUDATOOLKIT
pip install $flag --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag torch-geometric