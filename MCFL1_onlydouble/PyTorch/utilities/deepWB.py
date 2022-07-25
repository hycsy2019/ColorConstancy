"""
  Main function of MCFL (inference phase)

  Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""


import numpy as np
import torch
from torchvision import transforms
import utilities.utils as utls

from PIL import Image


def deep_wb(image, net_awb=None,device='cuda', s=656):
    image = Image.fromarray(image)  #array转成image

    image_resized = image.resize((round(image.width / max(image.size) * s), round(image.height / max(image.size) * s)))
    w, h = image_resized.size
    if w % 2 ** 4 == 0:
        new_size_w = w
    else:
        new_size_w = w + 2 ** 4 - w % 2 ** 4

    if h % 2 ** 4 == 0:
        new_size_h = h
    else:
        new_size_h = h + 2 ** 4 - h % 2 ** 4

    inSz = (new_size_w, new_size_h)
    if not ((w, h) == inSz):
        image_resized = image_resized.resize(inSz)

    # image_resized = image
    image = np.array(image)
    image_resized = np.array(image_resized)
    img = image_resized.transpose((2, 0, 1))
    img = img / 255
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    net_awb.eval()
    with torch.no_grad():   #强制之后的内容不进行计算图构建
        output_awb = net_awb(img)

    tf = transforms.Compose([   #将多种transform结合
        transforms.ToPILImage(),    #转换张量或ndarray到PIL图像
        transforms.ToTensor()   #转换PIL图像或numpy.ndarray到张量
    ])

    #torch.squeeze维度压缩。返回一个tensor（张量），其中 input 中大小为1的所有维都已删除
    output_awb = tf(torch.squeeze(output_awb.cpu()))
    output_awb = output_awb.squeeze().cpu().numpy()
    output_awb = output_awb.transpose((1, 2, 0))
    m_awb = utls.get_mapping_func(image_resized, output_awb)
    output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

    return output_awb