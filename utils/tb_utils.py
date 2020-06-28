import numpy as np
from torchvision.utils import make_grid
import torch
import yaml

def iter_scalar_metrics(writer,tag,loss,iter):
    """
    Writes the scalar into tensorboard
    tag: String representing the scalar
    loss: Loss value
    iter: step number
    """
    writer.add_scalar(tag,loss,iter)

def epoch_scalar_metrics(writer,tag,dictionary,epoch):
    """
    Write the scalar value into tensorboard
    tag: String representing the scalar
    dictionary: dictionary of stuff to be written
    epoch:epoch number
    """
    writer.add_scalars(tag,dictionary,epoch)

def image_writer(writer,tag,images,epoch):
    """
    write the image into a grid in tensorboard
    tag: the tag for images
    images: can be a tensor, numpy array or PIL object
    epoch: epoch number
    """
    img_grid = make_grid(images,nrow=8,normalize=False)
    writer.add_image(tag,img_grid,epoch)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.axis('off')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')