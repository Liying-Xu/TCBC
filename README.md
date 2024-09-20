# TCBC
Xu, L., Li, H., Shen, H., Zhang, C., and Zhang, L.: A Thin Cloud Blind Correction Method Coupling a Physical Model with Unsupervised Deep Learning for Remote Sensing Imagery, ISPRS Journal of Photogrammetry and Remote Sensing, 218, 246-259, 2024.

# Requisites
Main dependent packages：
- Python 3.6
- imageio==2.15.0
- libtiff==0.4.2
- matplotlib==3.3.4
- numpy==1.19.5
- opencv-python==4.5.4.58
- pandas==1.1.5
- Pillow==8.4.0
- tifffile~=2020.9.3
- torch~=1.10.0+cu111
- torchvision~=0.10.0+cu111

Other package details could be found in 'requirements.txt' in this project.

# Application
You should prepared data first and modify the path root in 'unaligned_dataset.py'. 

Then, you need to run 'train.py' to train the model. 

Finally, you need to run 'test.py' to corrected thin cloudy images using both a trained model and the target cloudy images as input.


We provide the dataset and corresponding trained correction model for testing, which is available in BaiduNetdisk. Link: https://pan.baidu.com/s/1EKyrVOwVrRZTDcm-yBiWqg?pwd=tcbc key：tcbc 

All images are top-of-atmosphere (TOA) reflectance data.
- Cloud set: The cloud images synthesized from cirrus band images.
- Ground set: The clear images acquired from real world.
- Thin cloudy set: The thin cloudy images acquired from real world.

When correcting your own data, we recommend rebuilding the dataset and retraining the correction model. The cloud set remains unchanged. The target thin cloudy images should be included in the thin cloudy set. The more similar the surface information of the ground set to that of the thin cloudy set, the better the correction results will be.

# Acknowledgments
Our code is inspired by [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [SPAGAN](https://github.com/Penn000/SpA-GAN_for_cloud_removal), and [YTMT](https://github.com/mingcv/YTMT-Strategy).
