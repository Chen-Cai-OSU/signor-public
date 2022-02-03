#!/usr/bin/env bash


CUDA='cu110' #'cu102'
CUDATOOLKIT='11.0'
flag='--ignore-installed --no-cache-dir'

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
#conda install cudatoolkit=$CUDATOOLKIT
pip install $flag --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install $flag torch-geometric