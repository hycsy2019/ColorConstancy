"""
 Training
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from arch import deep_wb_dct


try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = True
except ImportError:
    use_tb = False

from utilities.dataset import BasicDataset
from utilities.loss_func import mae_loss

from torch.utils.data import DataLoader, random_split

from PIL import Image
import matplotlib.pyplot as plt

def showImage(img):
    #显示3*h*w格式图片（非归一化）
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype(np.uint8))
    plt.imshow(img)
    plt.show()

def train_net(net,
              device,
              epochs=110,
              batch_size=32,
              lr=0.0001,
              val_percent=0.1,
              lrdf=0.5,
              lrdp=25,
              fold=0,
              chkpointperiod=1,
              trimages=12000,
              patchsz=128,
              patchnum=4,
              validationFrequency=4,
              dir_img='../dataset',
              save_cp=True):
    ### load modle
    # check = torch.load('/home/lcx/deepwhite/PyTorch/checkpoints_3_11/deep_WB_epoch134.pth')
    # start_epoch = check['epoch'] + 1
    # net.load_state_dict(check['model'])
    # # optimizer.load_state_dict(check['optimizer'])
    # print("=> loaded checkpoint (epoch {})".format(check['epoch']))

    dir_checkpoint = f'checkpoints_{fold}_11/'  #加f后可以在字符串里面使用用花括号括起来的变量和表达式
    #建立数据集
    dataset = BasicDataset(dir_img, fold=fold, patch_size=patchsz, patch_num_per_image=patchnum, max_trdata=trimages)
    n_val = int(len(dataset) * val_percent) #验证集数量
    n_train = len(dataset) - n_val  #训练集数量
    train, val = random_split(dataset, [n_train, n_val])    #划分

    #一般我们实现一个datasets对象，传入到dataloader中；然后内部使用yeild返回每一次batch的数据
    #num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
    #pin_memory(bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存中.
    #shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
    #drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
    #如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_FOLD3_BS_11_{batch_size}') #写入事件
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Patches/image:   {patchnum}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')
    #Adam算法基于训练数据迭代地更新神经网络权重
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    # optimizer.load_state_dict(check['optimizer'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)   #设置权重学习率衰减速度

    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        #tqdm进度条
        #total：总的项目数
        #desc：字符串，进度条左边描述文字
        #描述处理项目的文字,处理照片的话设置为'img' ,例如为 100 img/s
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                if batch:
                    #print(batch)
                    imgs_ = batch['image']  #色偏图像
                    awb_gt_ = batch['gt-AWB']   #原色图像
                    t_gt_ = batch['gt-T']
                    s_gt_ = batch['gt-S']
                    assert imgs_.shape[1] == net.n_channels * patchnum, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                        'the images are loaded correctly.'

                    assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                        'the images are loaded correctly.'

                    # assert t_gt_.shape[1] == net.n_channels * patchnum, \
                    #     f'Network has been defined with {net.n_channels} input channels, ' \
                    #     f'but loaded Tungsten WB GT images have {t_gt_.shape[1] / patchnum} channels. Please check that ' \
                    #     'the images are loaded correctly.'
                    #
                    # assert s_gt_.shape[1] == net.n_channels * patchnum, \
                    #     f'Network has been defined with {net.n_channels} input channels, ' \
                    #     f'but loaded Shade WB GT images have {s_gt_.shape[1] / patchnum} channels. Please check that ' \
                    #     'the images are loaded correctly.'

                    for j in range(patchnum):
                        imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                        awb_gt = awb_gt_[:, (j * 3): 3 + (j * 3), :, :]
                        t_gt = t_gt_[:, (j * 3): 3 + (j * 3), :, :]
                        s_gt = s_gt_[:, (j * 3): 3 + (j * 3), :, :]

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        awb_gt = awb_gt.to(device=device, dtype=torch.float32)
                        t_gt = t_gt.to(device=device, dtype=torch.float32)
                        s_gt = s_gt.to(device=device, dtype=torch.float32)

                        imgs_pred = net(imgs)


                        # loss = Angular_loss.compute(imgs_pred, torch.cat((awb_gt, t_gt, s_gt), dim=1))

                        #计算目标值与预测值之差绝对值和的均值
                        #torch.cat在给定维度上对输入的张量序列seq 进行连接操作
                        # print("维度：")
                        # print(awb_gt.size())
                        # print(torch.cat((awb_gt, t_gt, s_gt), dim=1).size())
                        loss = mae_loss.compute(imgs_pred, awb_gt)
                        epoch_loss.append(loss.item())

                        if use_tb:
                            writer.add_scalar('Loss/train', loss.item(), global_step)

                        pbar.set_postfix(**{'loss (batch)': loss.item()})#进度条后缀文字
                        optimizer.zero_grad()   #清空梯度
                        loss.backward()     #反向传播，计算当前梯度
                        optimizer.step()    #根据梯度更新网络参数
                        pbar.update(np.ceil(imgs.shape[0] / patchnum))  #更新进度条
                        global_step += 1
        with open('loss.txt', 'a') as f1:
            f1.write(str(epoch)+' '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
        #print(epoch_loss)


        #每validationFrequency个epoch验证一次
        if (epoch + 1) % validationFrequency == 0:
            if use_tb:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score = vald_net(net, val_loader, device)   #验证集MAE
            logging.info('Validation MAE: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result-awb', imgs_pred[:, :3, :, :], global_step)
                writer.add_images('result-t', imgs_pred[:, 3:6, :, :], global_step)
                writer.add_images('result-s', imgs_pred[:, 6:, :, :], global_step)
                writer.add_images('GT_awb', awb_gt, global_step)
                writer.add_images('GT-t', t_gt, global_step)
                writer.add_images('GT-s', s_gt, global_step)

        scheduler.step()#在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。

        #记录检查点
        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            checkpoint = {
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, dir_checkpoint + f'deep_WB_epoch{epoch+1}.pth')   #保存模型

            # torch.save(net.state_dict(), dir_checkpoint + f'deep_WB_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')
            # torch.save(net.state_dict(), dir_checkpoint + f'deep_WB_epoch{epoch + 1}.pth')
            # logging.info(f'Checkpoint {epoch + 1} saved!')

    #保存模型
    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'net_awb.pth')
    logging.info('Saved trained model!')
    # logging.info('Saving each auto-encoder model separately')
    # net_awb, net_t, net_s = splitter.splitNetworks(net)
    # torch.save(net_awb.state_dict(), 'models/' + 'net_awb_A.pth')
    # torch.save(net_t.state_dict(), 'models/' + 'net_t_A.pth')
    # torch.save(net_s.state_dict(), 'models/' + 'net_s_A.pth')
    # logging.info('Saved trained models!')
    if use_tb:
        writer.close()
    logging.info('End of training')


def vald_net(net, loader, device):
    """Evaluation using MAE"""
    net.eval()
    n_val = len(loader) + 1
    mae = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_ = batch['image']
            awb_gt_ = batch['gt-AWB']
            # t_gt_ = batch['gt-T']
            # s_gt_ = batch['gt-S']
            patchnum = imgs_.shape[1] / 3
            assert imgs_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            # assert t_gt_.shape[1] == net.n_channels * patchnum, \
            #     f'Network has been defined with {net.n_channels} input channels, ' \
            #     f'but loaded Tungsten WB GT images have {t_gt_.shape[1] / patchnum} channels. Please check that ' \
            #     'the images are loaded correctly.'
            #
            # assert s_gt_.shape[1] == net.n_channels * patchnum, \
            #     f'Network has been defined with {net.n_channels} input channels, ' \
            #     f'but loaded Shade WB GT images have {s_gt_.shape[1] / patchnum} channels. Please check that ' \
            #     'the images are loaded correctly.'

            imgs = imgs_[:, 0:3, :, :]
            awb_gt = awb_gt_[:, 0:3, :, :]
            # t_gt = t_gt_[:, 0:3, :, :]
            # s_gt = s_gt_[:, 0:3, :, :]
            imgs = imgs.to(device=device, dtype=torch.float32)
            awb_gt = awb_gt.to(device=device, dtype=torch.float32)
            # t_gt = t_gt.to(device=device, dtype=torch.float32)
            # s_gt = s_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():

                imgs_pred = net(imgs)

                # loss = Angular_loss.compute(imgs_pred, torch.cat((awb_gt, t_gt, s_gt), dim=1))
                # loss = mae_loss.compute(imgs_pred, torch.cat((awb_gt, t_gt, s_gt), dim=1))
                loss = mae_loss.compute(imgs_pred, awb_gt)
                mae = mae + loss

            pbar.update(np.ceil(imgs.shape[0] / patchnum))

    net.train()
    return mae / n_val


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')

    #一个epoch ,表示： 所有的数据送入网络中， 完成了一次前向计算 + 反向传播的过程
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=450,
                        help='Number of epochs', dest='epochs')
    #每个batch 中训练样本的数量
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    #学习速率
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    #加载训练好的模型参数
    parser.add_argument('-l', '--load', dest='load', type=str, default="/home/csy/pycharm/MCFL_onlydouble/PyTorch/checkpoints_0_11/deep_WB_epoch105.pth",
                        help='Load model from a .pth file')
    #用作验证集的数据所占总数据的百分比
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    #使用验证集实施验证的频率。当等于1时代表每个epoch结束都验证一次
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', type=int, default=5,
                        help='Validation frequency.')
    #用来验证的fold
    parser.add_argument('-d', '--fold', dest='fold', type=int, default=0,
                        help='Testing fold to be excluded. Use --fold 0 to use all Set1 training data')
    #每张图片的patch数
    parser.add_argument('-p', '--patches-per-image', dest='patchnum', type=int, default=4,
                        help='Number of training patches per image')
    #训练patch的大小
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=128,
                        help='Size of training patch')
    #训练图片数
    parser.add_argument('-t', '--num_training_images', dest='trimages', type=int, default=13333,
                        help='Number of training images. Use --num_training_images 0 to use all training images')
    #保存一个检查点间隔的epoch数
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                        help='Number of epochs to save a checkpoint')
    #学习率衰减速度
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf', type=float, default=0.5,
                        help='Learning rate drop factor')
    #学习率多长时间衰减一次
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp', type=int, default=25,
                        help='Learning rate drop period')
    #训练图像所在目录
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='/dataset/lcx/set1_all1/',
                        help='Training image directory')

    return parser.parse_args()


if __name__ == '__main__':
    #配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of Deep White-Balance Editing')
    #读取运行参数
    args = get_args()
    #统一计算设备架构（Compute Unified Device Architecture, CUDA），GPU编程接口
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # CUDA_VISIBLE_DEVICE = 1
    #初始化FIPE、SIG、CDCA及网络结构，FIPE Z字读取11个低频单位
    net = deep_wb_dct.deepWBnet(fre_sel_method1='low11',fre_sel_method2='low2')

    if args.load:   #加载模型
        checkpoint=torch.load(args.load, map_location=device)
        net.load_state_dict(    #将预训练的参数权重加载到新的模型之中
            checkpoint['model']        )
    logging.info(f'Model loaded from {args.load}')
    #
    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'net_loss.pth')
    net.to(device=device)   #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行

    for tag, value in net.named_parameters():   #将网络参数名中的.换成/
        tag = tag.replace('.', '/')

    try:
        #训练模型
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrdf=args.lrdf,
                  lrdp=args.lrdp,
                  device=device,
                  fold=args.fold,
                  chkpointperiod=args.chkpointperiod,
                  trimages=args.trimages,
                  val_percent=args.val / 100,
                  validationFrequency=args.val_frq,
                  patchsz=args.patchsz,
                  patchnum=args.patchnum,
                  dir_img=args.trdir
                  )
    except KeyboardInterrupt:   #中断训练，保存模型
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
