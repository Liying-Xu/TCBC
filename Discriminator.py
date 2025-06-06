import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
    # initializers
    def __init__(self, d, in_channels):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        #x = torch.cat([input, label], 1) pix2pix的判别器是比较是否为真实数据对，故要联结
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)),0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)),0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

