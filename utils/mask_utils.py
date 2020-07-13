import numpy as np
from torch.utils.data import Dataset
import skimage.transform as skt
import nibabel as nib
from glob import glob
from utils import image_utils

class load_data(Dataset):
    """Load normal images for reconstruction based fine tuning
    """
    def __init__(self,path,resize):
        self.path = path
        self.resize = resize
        self.data = sorted(glob(self.path+'*.nii.gz',recursive=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        file = self.data[index]
        filename = file.split('/')[-1]
        filename = filename.split('.')[0]
        img = nib.load(file)
        image = img.get_data()
        image = skt.resize(image,self.resize)
        image = image_utils.to_tensor(image)
        return image,filename

def save_masks(array,filename,path):
    np.save(path+filename+'_mask',array)

def normalise_mask(array):
    narray = array[:] - np.min(array) / (np.max(array) - np.min(array))
    return narray