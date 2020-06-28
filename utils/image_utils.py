import abc
from warnings import warn

import numpy as np
import torchvision.transforms as transforms
import torch
import skimage.util as skutil
import skimage.transform as skt
import sklearn.preprocessing as skp

def random_bbox(image, margin, patchsize):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = image.shape[0]
        img_width = image.shape[1]
        height = patchsize[0]
        width = patchsize[1]
        ver_margin = margin[0]
        hor_margin = margin[1]
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

def square_mask(image, margin, patchsize):
    """Generate mask tensor from bbox.
    Args:
    bbox: configuration tuple, (top, left, height, width)
    config: Config should have configuration including IMG_SHAPES,
    MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
    image shape inputted with just mask
    No ------tf.Tensor: output with shape [1, H, W, 1]
    """
    bboxs = []
    # for i in range(times):
    bbox = random_bbox(image, margin, patchsize)
    bboxs.append(bbox)
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros((height, width), np.float32)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        mask = np.expand_dims(mask,axis=2)
        mask = Resize(mask,(128,128))
        mask = np.expand_dims(mask,axis=2)
        mask = np.transpose(mask,(2,0,1))
        mask = torch.from_numpy(mask)
    return mask.float() #mask.reshape((1, ) + mask.shape).astype(np.float32)

def RandomHorizontalFlip(image):
    if np.random.uniform() < 0.5:
        image = image[:, ::-1]
    return image

def RandomCrop(image,output_size):
    image = image[:,:,0]
    shape = image.shape[0]
    width = shape - output_size
    cropped = skutil.crop(image,width)
    return cropped

def Resize(image,output_size):
    image = image[:,:,0]
    shape = image.shape[0]
    resized = skt.resize(image,output_size,order=3, mode='reflect')
    return resized

def to_tensor(image):
    img = torch.from_numpy(np.ascontiguousarray(image.transpose((2, 0, 1))))
    if isinstance(img, torch.ByteTensor) or img.dtype==torch.uint8:
        return img.float().div(255) #for normalisation of images
    else:
        return img

def normalise(image):
    image = skp.normalize(image)
    return image