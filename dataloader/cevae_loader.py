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
    def __init__(self,path,patchsize,margin,transforms = None,mask=False):
        self.path = path
        self.dataset = glob(path+'*.nii.gz',recursive=True)
        self.patchsize = patchsize
        self.transforms = transforms
        self.margin = margin
        self.mask = mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = nib.load(self.dataset[index])
        x = image.get_data()
        
        mask = square_mask(x,self.margin,self.patchsize)
#         if(self.mask == True):
#             x = square_mask(x,self.margin,self.patchsize)
        x = RandomHorizontalFlip(x)
        x = Resize(x,(128,128))
        x = normalise(x)
        x = np.expand_dims(x,axis=2)
        x = to_tensor(x).float()
        masked_image = torch.where(mask !=0,mask,x)
        return x ,masked_image