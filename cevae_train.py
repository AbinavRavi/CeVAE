from network import enc_dec
from dataloader import image_loader
from utils import loss_fn, tb_utils, utils
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from datetime import date

config = utils.read_config('./config/cevae_config.yml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = config['seed']
## Load the dataloader configurations
path = config['train']['dataloader']['path']
resize = tuple(config['train']['dataloader']['resize'])
patch_size = tuple(config['train']['dataloader']['patchsize'])
margin = tuple(config['train']['dataloader']['margin'])
batch_size = config['train']['dataloader']['batch']
num_workers = config['train']['dataloader']['num_workers']
## Load the model training configurations
log_path = config['train']['model']['log_path']
save_path = config['train']['model']['save_path']
h_size = config['train']['model']['hidden_dim']
input_size = config['train']['model']['input_size']
z_dim = config['train']['model']['z_dim']
lamda = torch.tensor(config['train']['model']['lamda'])
beta = torch.tensor(config['train']['model']['beta'])
## Load the optimizer paramaters
learning_rate = config['train']['optimizer']['lr']
epochs = config['train']['optimizer']['epochs']
weight_decay = config['train']['optimizer']['weight_decay']

##Validation loop
def validation(model,dataloader,writer,device,beta):
    
