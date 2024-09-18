import torch.nn as nn
import numpy as np
import os
import torch
import random
import pandas as pd
from torch.autograd import Variable
from PIL import Image
from torch import nn,optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from libtiff import TIFF
#from scipy import misc
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from PIL import Image
from torchvision import transforms as tfs



class ImageDataLoader():
    def __init__(self,data_path,gt_path,unaligned=False, mode='train',shuffle=False,pre_load=False):
        self.data_path=data_path
        self.gt_path=gt_path
        self.pre_load=pre_load
        self.data_files = [filename for filename in os.listdir(data_path)\
                        if os.path.isfile(os.path.join(data_path,filename))]
        self.gt_files= [filename for filename in os.listdir(gt_path)\
                        if os.path.isfile(os.path.join(gt_path,filename))]
        self.data_files.sort()
        self.gt_files.sort()
        self.shuffle=shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples=len(self.data_files)
        self.blob_list={}
        self.id_list=list(range(0,self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx1 = 0
            for fname1 in self.data_files:
                blob={}
                tif1 = TIFF.open(self.data_path+"/"+fname1, mode="r")
                for im1 in list(tif1.iter_images()):
                    im1 = im1.astype(np.float32, copy=False)

                    blob['data']=im1
                path1 = os.path.splitext(fname1)[0]
                path2 = os.path.splitext(fname1)[1]
                path1 = path1.replace('c', 'f')
                fname2 = path1 + path2  #"Output" + path1 + path2
                path3=self.gt_path+"\\"+fname2
                print(path3)
                tif2 = TIFF.open(path3, mode="r")
                for im2 in list(tif2.iter_images()):
                    im2=im2.astype(np.float32, copy=False)

                    blob['gt']=im2
                blob['fname1'] = fname1
                blob['fname2'] = fname2
                self.blob_list[idx1] = blob
                idx1=idx1+1
                if idx1 % 100 == 0:
                    print('Loaded(Train-cloudly) ', idx1, '/', self.num_samples, 'files')
                    print('\n')
                    print('Loaded(Train-free_cloudly) ', idx1, '/', self.num_samples, 'files')


    #__iter__函数是一个迭代函数，所以该类实际上是一个迭代器，需要用for...in 来进行调用
    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files=self.data_files
        id_list=self.id_list

        for idx in id_list:
            if self.pre_load:
                blob=self.blob_list[idx]
                blob['idx']=idx
            else:
                fname1=files[idx]
                path1 = os.path.splitext(fname1)[0]
                path2 = os.path.splitext(fname1)[1]
                path1 = path1.replace('c', 'f')
                fname2 =path1[5:] + path2 #"Output" + path1[5:] + path2
                tif1 = TIFF.open(self.data_path + "/" + fname1, mode="r")
                for im1 in list(tif1.iter_images()):
                    im1 = im1.astype(np.float32, copy=False)
                tif2 = TIFF.open(self.gt_path + "/" + fname2, mode="r")
                for im2 in list(tif2.iter_images()):
                    im2 = im2.astype(np.float32, copy=False)
                blob = {}
                blob['data']=im1
                blob['gt']=im2
                blob['fname1'] = fname1
                blob['fname2'] = fname2

            yield blob

    def get_num_samples(self):
        return self.num_samples


    def __len__(self):
        return max(len(self.data_files),len(self.gt_files))

#
# datapath="E:/JY/Dataset1/Train/Input"
# gtpath="E:/JY/Dataset1/Train/Output"
# data=ImageDataLoader(datapath,gtpath,shuffle=True,pre_load=True)
# im_data1=[]
# fname1=[]
# fname2=[]
# for blob in data:
#     fname1.append(blob['fname1'])
#     fname2.append(blob['fname2'])
#
# print(fname1)
# print(fname2)
#     fname1.append(blob['fname1'])
# #存储TIFF文件# print(im_data1[0][0,0])
# out_tiff = TIFF.open("E:/JY/2.tif", mode = 'w')
# out_tiff.write_image(im_data1[0], compression = None, write_rgb = True)


