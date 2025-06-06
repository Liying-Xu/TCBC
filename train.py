import os, time, pickle, util, argparse
from unaligned_dataset import UnalignedDataset
from option import TrainOptions
import datetime
import itertools
from FSNet import FSNet
from Discriminator import discriminator
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# parameter saving path
opt = TrainOptions().parse()
opt.log_path = os.path.join(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.txt')
#opt.dataset='06'
# results save path
root = opt.dataset + '_' + opt.save_root + '/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)

# data_loader
Udataset = torch.utils.data.DataLoader(
            UnalignedDataset(opt),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))  # create a dataset given opt.dataset_mode and other options

G = FSNet()
G.cuda()

D = discriminator(d=32, in_channels=10)
D.cuda()

# loss
L1_loss = nn.L1Loss().cuda()
L2_loss = nn.MSELoss().cuda()
# Adam optimizer
G_optimizer = torch.optim.Adam(itertools.chain(G.parameters()), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=util.LambdaLR(opt.train_epoch, opt.epoch,
                                                                                        opt.decay_epoch).step)

D_optimizer = torch.optim.Adam(itertools.chain(D.parameters()), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda=util.LambdaLR(opt.train_epoch, opt.epoch,
                                                                                        opt.decay_epoch).step)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_batch():
    G.train()
    D.train()
    # results save folder
    for epoch in range(opt.train_epoch):
        epoch_start_time = time.time()
        for batch in enumerate(Udataset):
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            ######################################################################### 图像
            n_count=batch[0]#get the number of batch
            data=batch[1]
            ######################################################################### 图像

            x_ = torch.as_tensor(data['A']).permute(0, 3, 1, 2).cuda()  # 待校正
            y_ = torch.as_tensor(data['B']).permute(0, 3, 1, 2).cuda()  # 参考地表图像
            z_ = torch.as_tensor(data['C']).permute(0, 3, 1, 2).cuda()  # 参考云图像

            Gx = G(x_)
            Gc_x = Gx[0]  # 云分量
            Gg_x = Gx[1]  # 地表分量


            Gz = G(z_)
            Gc_z = Gz[0]  # 同z_

            Gy = G(y_)
            Gg_y = Gy[1]  # 同y_

            yz_ = z_ + y_
            Gyz = G(yz_)
            Gc_yz = Gyz[0]
            Gg_yz = Gyz[1]
            Ga_yz = Gc_yz + Gg_yz
            ##########################################################################
            D_real = D(torch.cat((yz_, y_), 1))
            D_real_loss = L2_loss(D_real, torch.ones(D_real.size()).cuda())
            D_fake = D(torch.cat((x_, Gg_x), 1))
            D_fake_loss = L2_loss(D_fake, torch.zeros(D_fake.size()).cuda())
            lossD = D_real_loss + D_fake_loss
            lossD.backward(retain_graph=True)
            ########################################################################### 生成器损失
            separationLoss = 0.8 * L1_loss(Gg_yz, y_) \
                             + 0.2 * L1_loss(Ga_yz, yz_)
            additionLoss = L1_loss(Ga_x, x_)
            identityLoss = 0.5 *L1_loss(Gg_y, y_)+ 0.5 * L1_loss(Gc_z, z_)


            ganLoss = L2_loss(D_fake, torch.ones(D_fake.size()).cuda())

            lossG = 10*separationLoss + 5*additionLoss + ganLoss + 5*identityLoss

            lossG.backward(retain_graph=True)  # 保留backward后的中间参数。
            ########################################################################### loss值更新
            G_loss = 0
            D_loss = 0
            addition_Loss = 0
            identity_Loss = 0
            separation_Loss = 0
            gan_Loss = 0
            G_loss += lossG.item()
            D_loss += lossD.item()
            addition_Loss += additionLoss.item()
            identity_Loss += identityLoss.item()
            separation_Loss += separationLoss.item()
            gan_Loss += ganLoss.item()
            ########################################################################### optimizer更新
            G_optimizer.step()
            D_optimizer.step()
            ########################################################################### 记录loss
            train_hist['G_loss'].append(G_loss)
            train_hist['D_loss'].append(D_loss)
            train_hist['Addition_Loss'].append(addition_Loss)
            train_hist['Identity_Loss'].append(identity_Loss)
            train_hist['Separation_Loss'].append(separation_Loss)
            train_hist['Gan_Loss'].append(gan_Loss)
        ############################################################################### epoch计时/打印loss
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        avg_log = '[%d/%d] - ptime: %.2f,G_loss: %.8f,D_loss: %.8f,Addition_loss:%.8f,Identity_Loss:%.8f,Separation_Loss:%.8f,GAN_Loss:%.8f' % (
            (epoch + 1), opt.train_epoch, per_epoch_ptime,
            (G_loss/opt.batch_size/n_count),
            (D_loss/opt.batch_size/n_count),
            (addition_Loss/opt.batch_size/n_count),
            (identity_Loss/opt.batch_size/n_count),
            (separation_Loss/opt.batch_size/n_count),
            (gan_Loss/opt.batch_size/n_count))
        print(avg_log)
        open(opt.log_path, 'a').write(avg_log + '\n')
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        ############################################################################### 存模型
        if (epoch + 1) % 5 == 0:
            torch.save(G.state_dict(), (root + model + 'G7_param_%d.pth' % (epoch + 1)))
            torch.save(D.state_dict(), (root + model + 'D7_param_%d.pth' % (epoch + 1)))
    util.show_train_hist(train_hist, save=True, path=root + model + '_7_train_hist.png')


if __name__ == '__main__':
    train_hist = {}
    train_hist['G_loss'] = []
    train_hist['D_loss'] = []
    train_hist['Addition_Loss'] = []
    train_hist['Identity_Loss'] = []
    train_hist['Separation_Loss'] = []
    train_hist['Gan_Loss'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    start_time = time.time()
    epoch=0
    train_batch()
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
    print("Training finish!... save training results")

    with open(root + model + '_7_train_hist.pth', 'wb') as f:
        pickle.dump(train_hist, f)