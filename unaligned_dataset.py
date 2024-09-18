import os.path
import random
import tifffile
import torch
from torch.nn import functional as F
import math

"""
含transform时，最终结果变化很小，且出现棋盘效应
def transform(img_torch):
    degree=random.randint(0,3)
    angle = (degree*90)*math.pi/180
    theta = torch.tensor([
        [math.cos(angle),math.sin(-angle),0],
        [math.sin(angle),math.cos(angle) ,0]], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = F.grid_sample(img_torch.unsqueeze(0), grid).squeeze(0)
    return output
"""
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)#append：拼接操作
    return images[:min(max_dataset_size, len(images))]
#########################################################################A:有云数据集 B:地表真实影像集
class UnalignedDataset():

    def __init__(self,opt):

        A_root = opt.dataroot +'\\'+ opt.phase +'\\'+ opt.dataset + '\cloudy'
        B_root = opt.dataroot +'\\'+ opt.phase +'\\'+ opt.dataset + '\\freecloudy'
        C_root = r'G:\XLY\DATA\XLY_DATA\gamma_cloud'

        self.dir_A = os.path.join(A_root)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(B_root)  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(C_root)  # create a path '/path/to/data/trainC'


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))  # load images from '/path/to/data/trainC'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)  # get the size of dataset C

    def __getitem__(self, index):


        A_path = self.A_paths[index % self.A_size] # make sure index is within then range
        """
        #配对:
            index_B = index % self.B_size
        #不配对: randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        """
        index_B = random.randint(0, self.B_size - 1)
        index_C = random.randint(0, self.C_size - 1)

        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]

        A = tifffile.imread(A_path).astype('float32')
        B = tifffile.imread(B_path).astype('float32')
        C = tifffile.imread(C_path).astype('float32')


        return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}

    def __len__(self):
        return max(self.A_size, self.B_size)#, self.C_size)
