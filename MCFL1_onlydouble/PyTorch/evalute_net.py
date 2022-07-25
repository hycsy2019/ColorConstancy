import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from arch import deep_wb_dct
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = True
except ImportError:
    use_tb = False

from utilities.dataset import BasicDataset
from utilities.crop_val import MultiEvalModule,MultiEvalModule_l,MultiEvalModule_ll,MultiEvalModule_lll
from utilities.loss_func import mae_loss,mse_loss,delta_e_loss

from torch.utils.data import DataLoader, random_split
import logging
from PIL import Image, ImageOps

def encode(hist):
    """ Generates a compacted feature of a given RGB-uv histogram tensor."""
    encoderBias = np.load('./models/encoderBias+.npy')
    encoderWeights = np.load('./models/encoderWeights+.npy')
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - encoderBias.transpose(),
                     encoderWeights)
    return feature


def error_evaluation(error_list,model_name,camera,type):
    es = np.array(error_list)
    es.sort()
    ae = np.array(es).astype(np.float32)

    x, y, z = np.percentile(ae, [25, 50, 75])
    Mean = np.mean(ae)
    Med = np.median(ae)
    Tri = (x+ 2 * y + z)/4
    T25 = np.mean(ae[:int(0.25 * len(ae))])
    L25 = np.mean(ae[int(0.75 * len(ae)):])

    print("Mean\tMedian\tTri\tBest 25%\tWorst 25%")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(Mean, Med, Tri, T25, L25))


def to_image(image):
    """ converts to PIL image """
    return Image.fromarray((image * 255).astype(np.uint8))

def vald_net(net, loader, device):
    """Evaluation using MAE"""

    n_val = len(loader) + 1
    print(n_val)
    #mse, IMG = [],  []
    mse,delta,IMG = [],[],[]


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        k=0
        for batch in loader:
            imgs_ = batch['image']
            awb_gt_ = batch['gt-AWB']

            patchnum = imgs_.shape[1] / 3
            assert imgs_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'


            imgs = imgs_[:, 0:3, :, :]
            awb_gt = awb_gt_[:, 0:3, :, :]

            b,c,h,w = imgs.size()
            crop_val = MultiEvalModule_lll(net,h,128,device)

            imgs = imgs.to(device=device, dtype=torch.float32)
            awb_gt = awb_gt.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                imgs_pred = crop_val(imgs)

                #show预测图像
                # imgs_pred1 = imgs_pred[0].cpu().numpy().transpose((1, 2, 0))
                # imgs_pred1 = to_image(imgs_pred1)
                # plt.imshow(imgs_pred1)
                # #plt.savefig('/home/csy/pycharm/MCFL1/test.png')
                # plt.show()

                loss_mse = mse_loss.compute(imgs_pred[:, 0:3, :, :]* 255, awb_gt[:, 0:3, :, :]* 255)
                loss_d = delta_e_loss.compute(imgs_pred[:, 0:3, :, :], awb_gt[:, 0:3, :, :])
                mse.append(loss_mse.cpu().numpy())
                delta.append(loss_d.cpu().numpy())
                pbar.update(np.ceil(k+1))

    mse = np.array(mse)
    delta = np.array(delta)
    IMG = np.array(IMG)

    error_evaluation(mse, args.model_name, args.camera, '_All_mse')
    error_evaluation(delta, args.model_name, args.camera, '_All_deltae')

    return mse,IMG,delta


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')

    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=128,
                        help='Size of training patch')

    parser.add_argument('--model_dir', '-m', default='/home/csy/pycharm/MCFL_onlydouble/PyTorch/models/',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='/home/csy/datasets/Cube+/',
                        help='Training image directory')
    parser.add_argument('-model_name', '--model_name', dest='model_name',
                        default='net_loss')
    parser.add_argument('-type', '--type', dest='type', default='test')  ## dc: 0
    parser.add_argument('-cam', '--camera', dest='camera', default='set1_all',
                        help='Testing camera')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    logging.info(f'Testing {args.camera} via {args.model_name}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    model_name = args.model_name

   ##### 载入net
   ##### 平常测网络精度的时候记得要把 dataset_cam_t.py 中的resize删掉
    net_awb = deep_wb_dct.deepWBnet(fre_sel_method1='low11',fre_sel_method2='low2',device=device)

    #加载模型
    logging.info("Loading model {}".format(
        os.path.join(args.model_dir,
                     model_name + '.pth')))
    net_awb.load_state_dict(
            torch.load(os.path.join(args.model_dir, model_name + '.pth'))
        )


    net_awb.to(device=device)
    net_awb.eval()  #测试模型

    ##### 载入数据nimgs_dir, gt_dir, patch_size=128, type='cube'
    val = BasicDataset(imgs_dir="/home/csy/datasets/mixed/")
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    #### 测试
    mse,IMG,delta = vald_net(net_awb, val_loader, device)
    # np.save('/home/lcx/deep_final/PyTorch/ERROR/' + args.model_name + '_' + args.camera+ '_uvpca.npy', IMG)
    np.save('/home/csy/pycharm/MCFL_onlydouble/PyTorch/ERROR/' + 'awb_'+model_name+ '_MSE', mse)
    np.save('/home/csy/pycharm/MCFL_onlydouble/PyTorch/ERROR/' + 'awb_'+model_name+ '_deltaE', delta)
    print('d')



