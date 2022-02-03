# Created at 2020-04-14
# Summary: pytorch utils

def model_device(model):
    """
    get the device location of a model
    https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/14
    """
    return next(model.parameters()).device
