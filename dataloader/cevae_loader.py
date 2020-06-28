import numpy as np
import nibabel as nib
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset
from utils.image_utils import *
from glob import glob

class cevae(Dataset):
    def __init__(self,path,patchsize,margin,resize):
        self.path = path
        self.dataset = glob(path+'*.nii.gz',recursive=True)
        self.patchsize = patchsize
        self.margin = margin
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = nib.load(self.dataset[index])
        x = image.get_data()
        
        mask = square_mask(x,self.margin,self.patchsize)
        x = RandomHorizontalFlip(x)
        x = Resize(x,self.resize)
        x = normalise(x)
        x = np.expand_dims(x,axis=2)
        x = to_tensor(x).float()
        masked_image = torch.where(mask !=0,mask,x)
        return x ,masked_image