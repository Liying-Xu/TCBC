import torch,  os
from option import TestOptions
import tifffile
from libtiff import TIFF
import numpy as np
from FSNet import FSNet

opt = TestOptions().parse()
print(opt)
opt.dataset='soil'
root=r"G:\TC-BC\results"
if not os.path.isdir(root+r'test_results'):
    os.mkdir(root+r'test_results')

Gg = FSNet()
modelroot=root +opt.dataset+ r'_result\\'+opt.dataset+'_G_param_%d.pth'%(opt.n_epoch)
dataroot=(r'G:\\'+opt.dataset+r'\\TEST')

print('modelroot:',modelroot)
print('dataroot:',dataroot)

Gg.load_state_dict(torch.load(modelroot))

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

dir_A = os.path.join(dataroot)#create a path '/path/to/data/trainA'#
A_path = sorted(make_dataset(dir_A, opt.max_dataset_size))

with torch.no_grad():
    for i in range (len(A_path)):
        x_name= os.path.basename(A_path[i])
        x_ = tifffile.imread(A_path[i]).astype('float32')
        x_ = torch.from_numpy(x_).unsqueeze(0)#.permute(0, 3, 1, 2)
        Gg_x=Gg(x_)
        Gg_images = (Gg_x[1]).cpu().detach().numpy().squeeze().astype(np.float32)
        Gc_images = (Gg_x[0]).cpu().detach().numpy().squeeze().astype(np.float32)
        Ga_images = (Gg_x[2]).cpu().detach().numpy().squeeze().astype(np.float32)

        out_tiff_g = TIFF.open(root+'test_results'+'\\Gg_%d_%s' % (opt.n_epoch,x_name), mode='w')
        out_tiff_g.write_image(Gg_images, compression=None, write_rgb=True)
        out_tiff_c = TIFF.open(root+'test_results'+'\\Gc_%d_%s' % (opt.n_epoch,x_name), mode='w')
        out_tiff_c.write_image(Gc_images, compression=None, write_rgb=True)
        out_tiff_a = TIFF.open(root +'test_results' + '\\Ga_%d_%s' % (opt.n_epoch, x_name), mode='w')
        out_tiff_a.write_image(Ga_images, compression=None, write_rgb=True)
print('%d images generation complete!' % n)
