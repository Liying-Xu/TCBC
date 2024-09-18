import glob
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
import torch
import random

patch_size, stride = 256, 256
aug_times = 1
B =32

def gen_patches(x_path = ''):
    x = tiff.imread(x_path)
    x = np.array(x).astype(np.float32)
    C,H,W=x.shape
    x_patch = []
    for i in range(0, H-patch_size+1, stride):
        for j in range(0, W-patch_size+1, stride):
            x_ = x[:, i: i + patch_size, j: j + patch_size]
            x_patch.append(x_)

    return x_patch
def gen_patchesc(x_path = ''):
    x = tiff.imread(x_path).transpose(2,0,1)
    x = np.array(x).astype(np.float32)
    C,H,W=x.shape
    x_patch = []
    for i in range(0, H-patch_size+1, stride):
        for j in range(0, W-patch_size+1, stride):
            x_ = x[:, i: i + patch_size, j: j + patch_size]
            x_patch.append(x_)

    return x_patch
def data_generator(x_path='', y_path='',z_path =''):
    x_data = []
    y_data = []
    z_data = []
    x_list = glob.glob(x_path + '/*.tif')
    y_list = glob.glob(y_path + '/*.tif')
    z_list = glob.glob(z_path + '/*.tif')
    for i in range(len(x_list)):
        x_patch= gen_patches(x_list[i])
        for x_patch_ in x_patch:
            x_data.append(x_patch_)
    print(len(x_data),'finish load x')
    x_data1 = np.array(x_data, dtype='float32')

    for i in range(len(y_list)):
        y_patch= gen_patches(y_list[i])
        for y_patch_ in y_patch:
            y_data.append(y_patch_)
    print(len(y_data),'finish load y')
    y_data1 = np.array(y_data, dtype='float32')

    for i in range(len(z_list)):
        z_patch = gen_patchesc(z_list[i])
        for z_patch_ in z_patch:
            z_data.append(z_patch_)
    print(len(z_data),'finish load z')
    z_data1 = np.array(z_data, dtype='float32')
    discard_x = len(x_data1) - len(x_data1) // B * B
    discard_y = len(y_data1) - len(y_data1) // B * B
    discard_z = len(z_data1) - len(z_data1) // B * B
    x_data2 = np.delete(x_data1, range(discard_x), axis=0)
    y_data2 = np.delete(y_data1, range(discard_y), axis=0)
    z_data2 = np.delete(z_data1, range(discard_z), axis=0)

    print('^_^-training data finished-^_^')
    x_data3 = torch.from_numpy(x_data2)
    y_data3 = torch.from_numpy(y_data2)
    z_data3 = torch.from_numpy(z_data2)
    return x_data3, y_data3, z_data3

class train_dataset(Dataset):

    def __init__(self,x_data3, y_data3, z_data3):
        super(train_dataset,self).__init__()
        self.x_data = x_data3
        self.y_data = y_data3
        self.z_data = z_data3
        self.x_size=len(x_data3)
        self.y_size = len(y_data3)
        self.z_size = len(z_data3)

    def __getitem__(self, index):
        x_data_b = self.x_data[index % self.x_size]
        index_y = random.randint(0, self.y_size- 1)
        y_data_b = self.y_data[index_y % self.y_size]
        index_z = random.randint(0, self.z_size)
        z_data_b = self.z_data[index_z % self.z_size]

        return x_data_b.float(), y_data_b.float(), z_data_b.float()

    def __len__(self):
        return self.x_data.size(0)


