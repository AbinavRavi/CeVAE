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
import pdb

config = utils.read_config('./config/cevae_config.yml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = config['seed']
utils.set_seed(seed)
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
def validation(model,dataloader,writer,device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for idx,(data,masked_data) in enumerate(tqdm(dataloader,desc='val_iter',leave=False)):
            data, masked_data = data.to(device), masked_data.to(device)
            rec_vae,mu, std = model(data)
            rec_ce,_,_ = model(masked_data)
            loss,loss_vae = loss_fn.criterion(rec_vae,data,rec_ce,masked_data,mu,std)
            val_loss.append(loss.item())
            tb_utils.iter_scalar_metrics(writer,'Itr/Validation',loss.item(),i*len(dataloader)+idx)
            input_data = data.detach().cpu()
            vae_image = rec_vae.detach().cpu()
            ce_image = rec_ce.detach().cpu()
        return np.array(val_loss).mean(),input_data ,vae_image,ce_image

def train(model,train_loader,writer,device,optimizer):
    train_loss = []
    model.train()
    for idx,(data,masked_image) in enumerate(tqdm(train_loader,desc='train_iter',leave=False)):
        data,masked_image = data.to(device),masked_image.to(device)
        optimizer.zero_grad()
        rec_vae,mu,std = model(data)
        rec_ce, _, _ = model(masked_image)
        loss, loss_ce = loss_fn.criterion(rec_vae,data,rec_ce,masked_image,mu,std)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        tb_utils.iter_scalar_metrics(writer,'Itr/Train',loss.item(),i*len(train_loader)+idx)
    return np.array(train_loss).mean()

train_loader,val_loader = image_loader.cevae_batch(path,patch_size,margin,resize,batch_size=batch_size,num_workers=num_workers)

model = enc_dec.VAE(input_size, h_size, z_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer,threshold=0.0001,eps=1e-4)

writer = SummaryWriter(f'{log_path}{date.today()}_multi_task_learning_{lamda}_{learning_rate}_{batch_size}')
##training loop
for i in range(epochs):
    train_losses = train(deepcopy(model),train_loader,writer,device,optimizer)
    val_losses, idata,recon_vae, recon_ce = validation(deepcopy(model),val_loader,writer,device)
    tb_utils.image_writer(writer,'Epoch/Input_data',idata,i)
    tb_utils.image_writer(writer,'Epoch/recon_vae',recon_vae,i)
    tb_utils.image_writer(writer,'Epoch/recon_ce',recon_ce,i)
    dic = {'train':train_losses,'val':val_losses}
    tb_utils.epoch_scalar_metrics(writer,'Epoch/loss',dic,i)
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(train_losses),'\t','valloss:{}'.format(val_losses))
    if((i+1)%4 == 0 and (i+1)>30):
        torch.save(model,f'{save_path}CeVAE_V1_{batch_size}_{learning_rate}_{i+1}.pt')