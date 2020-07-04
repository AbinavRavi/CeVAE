from torch.utils.data import random_split
from torch.utils.data import DataLoader 
from dataloader.cevae_loader import cevae

def cevae_batch(path,margin,patchsize,resize,batch_size=128,num_workers=1,split = 0.2):
    dataset = cevae(path,patchsize,margin,resize)
    val_size = int(split*len(dataset))
    train_size = int((1-split)*len(dataset))
    train_ds, val_ds = random_split(dataset,[train_size,val_size])
    return DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers), DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)