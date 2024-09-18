# TCBC
thin cloud blind correction method

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

# TCBC
+You should prepared data first and modify the path root in 'unaligned_dataset.py'. 
+If you want to train the model by yourself, you need to run 'train.py'. 
+If you want to corrected thin cloudy images, you need to run 'test.py' and provide both a trained model and the target cloudy images as input.

+We provide the UBCSet and corresponding trained correction model for testing, which can be downloaded as follows.
+When correcting your own data, we recommend rebuilding the dataset and retraining the correction model. (The cloud set remains unchanged. The target thin cloudy images should be included in the thin cloudy set. The more similar the surface information of the ground set is to that of the thin cloudy set, the better the correction results will be.)

# UBCset
All images in UBCSet are top-of-atmosphere (TOA) reflectance data.

Cloud set: The cloud images synthesized from cirrus band images.
Ground set: The clear images acquired from real world.
Thin cloudy set: The thin cloudy images acquired from real world.

# Acknowledgments
Our code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
