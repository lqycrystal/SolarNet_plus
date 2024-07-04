from os.path import exists, join, isfile

import torch.utils.data as data
from os import listdir
from PIL import Image
import numpy as np
import torch
from skimage import io
import os
def bsd500(dest):

    if not exists(dest):
        print("dataset not exist ")
    return dest
class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).float()
            img = img.transpose(0, 1).transpose(0, 2)
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], 1)
            img = img.transpose(0, 1).transpose(0, 2).squeeze().contiguous()
        return img

def input_transform(crop_size):
        return ToTensor()

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
            label = label.transpose(0, 1).transpose(0, 2)
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous()
        return label
def input_transform2(crop_size):
    return LabelToLongTensor()


def get_training_set(dest,size, target_mode='seg', colordim=1):
    root_dir = bsd500(dest)
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir,target_mode,colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform2(size))


def get_test_set(dest,size, target_mode='seg', colordim=1):
    root_dir = bsd500(dest)
    test_dir = join(root_dir, "val")
    return DatasetFromFolder(test_dir,target_mode,colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform2(size))




def is_image_file(filename):
    return [name for name in listdir(filename) if isfile(join(filename, name))]


def load_img(filepath,colordim):
    
    if colordim==1:
        img = io.imread(filepath)

    else:

        img = io.imread(filepath)
        img=img[:,:,0:3]/255.0

    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_mode, colordim, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames =  is_image_file(join(image_dir,'seg'))
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.target_mode = target_mode
        self.colordim = colordim
        self.numFiles = len(self.image_filenames)
    def __getitem__(self, index):


        input = load_img(join(self.image_dir,'data',os.path.splitext(self.image_filenames[index])[0]+'.png'),self.colordim)

        
        target = load_img(join(self.image_dir,self.target_mode,self.image_filenames[index]),1)
        


        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return int(self.numFiles)

