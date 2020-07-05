import numpy as np
import torch
import yaml
import torch.nn as nn

def set_seed(seed = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def read_config(file):
    with open(file,'r') as stream:
        cxr_config = yaml.full_load(stream)
    return cxr_config

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)