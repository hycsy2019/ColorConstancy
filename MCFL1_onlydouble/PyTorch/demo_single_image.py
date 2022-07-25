"""
 Demo single image
"""

import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_dct
from utilities.deepWB import deep_wb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def get_args():
    # 构造一个参数解析器对象，description显示在使用命令和参数解释之间，一般用来描述程序的作用
    parser = argparse.ArgumentParser(description='Changing WB of an input image.')

    #前面带'-'的是可选参数，不带'-'的是位置参数
    #dest表示属性名
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename', dest='input',
                        default='example_images/R0001272.JPG')
    parser.add_argument('--output_dir', '-o', default='result_images',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--task', '-t', default='all',
                        help="Specify the required task: 'AWB', 'editing', or 'all'.", dest='task')
    parser.add_argument('--target_color_temp', '-tct', default=None, type=int,
                        help="Target color temperature [2850 - 7500]. If specified, the --task should be 'editing'",
                        dest='target_color_temp')
    parser.add_argument('--mxsize', '-S', default=656, type=int,
                        help="Max dim of input image to the network, the output will be saved in its original res.",
                        dest='S')
    parser.add_argument('--show', '-v', action='store_true', default=True,
                        help="Visualize the input and output images",
                        dest='show')    #show默认设为True
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')    #设置日志系统的级别、格式
    args = get_args()   #获取参数设置
    if args.device.lower() == 'cuda':   #？
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    fn = args.input   #输入图像的路径
    out_dir = args.out_dir  #输出图像的路径
    S = args.S      #输入图像的最大尺度，输出将保存在其原始分辨率
    target_color_temp = args.target_color_temp  #editing任务下设置的色温
    tosave = True      #是否保存图像

    #### loading model
    #加载模型，FIPE取低频11个单位，CDCA取低频18个单位
    net_awb = deep_wb_dct.deepWBnet(fre_sel_method1='low11', fre_sel_method2='low18')
    net_awb.load_state_dict(
        torch.load('models/net1.pth'
                   , map_location='cuda:0'))    #load_state_dict(state_dict, strict=True)给模型对象加载训练好的模型参数

    net_awb.to(device=device)
    net_awb.eval()

    #### input color cast rendered image
    logging.info("Processing image {} ...".format(fn))
    img = Image.open(fn)
    plt.imshow(img)
    plt.show()

    ### WB input image
    img = np.array(img)
    out_awb = deep_wb(img, net_awb=net_awb, device=device, s=S)
    out_awb = (out_awb * 255).astype('uint8')   # uint8是无符号八位整型，表示范围是[0, 255]的整数
    plt.imshow(out_awb)
    plt.show()

    ### to save
    if tosave:
        mpimg.imsave(os.path.join(out_dir, 'cor_wb1.png'),out_awb)




