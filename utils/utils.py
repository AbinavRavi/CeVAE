import numpy as np
import torch
import yaml

def set_seed(seed = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def read_config(file):
    with open(file,'r') as stream:
        cxr_config = yaml.full_load(stream)
    return cxr_config