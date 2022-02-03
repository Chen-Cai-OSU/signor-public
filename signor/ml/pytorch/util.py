import torch
import torch.nn as nn
import os

def basic_check():
    print(f'torch version: {torch.__version__}')  # PyTorch version
    if torch.cuda.is_available():
        print(torch.version.cuda)  # Corresponding CUDA version
        print(torch.backends.cudnn.version())  # Corresponding cuDNN version
        print(torch.cuda.get_device_name(0))  # GPU type
    else:
        print('cuda not available')


def set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def model_params(model, matcher = True):
    # https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778
    assert isinstance(model, nn.Module)
    params = dict()
    for name, param in model.named_parameters():
        criteria = True if matcher is True else matcher in name  #todo: better matcher
        if criteria:
            print('{:30}'.format(name), end='\t')
            print(param.data.size())
            params[name] = param
    return params

def set_gpu(dev = '0, 1'):
    os.environ['CUDA_VISIBLE_DEVICES'] = dev


if __name__ == '__main__':
    basic_check()
