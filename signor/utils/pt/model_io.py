# https://pytorch.org/tutorials/beginner/saving_loading_models.html

import torch
import shutil


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    """ An example
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)


    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    pass
