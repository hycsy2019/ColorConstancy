"""
 Loss function
"""

import torch
import pytorch_colors as colors
import torch.nn.functional as F

class mae_loss():

    @staticmethod
    def compute(output, target):
        # print(output.size())
        # print(target.size())
        # print(output.size(0))
        #output1=output.repeat(1,3,1,1)
        #print(output1.size())
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss


class mse_loss():

    @staticmethod
    def compute(output, target):
        output=torch.reshape(output,[-1,1])
        target=torch.reshape(target,[-1,1])
        loss=torch.sum(torch.pow((output-target),2))/output.shape[0]
        # loss = (torch.sum(torch.pow((output[:, 0, :, :] - target[:, 0, :, :]), 2) + torch.pow(
        #     (output[:, 1, :, :] - target[:, 1, :, :]), 2)
        #                  + torch.pow((output[:, 2, :, :] - target[:, 2, :, :]), 2)))/(output.size(2)*output.size(3))
        # kk = output.size(0)
        return loss

class delta_e_loss():

    @staticmethod
    def compute(output, target):
        b, c, h, w = output.size()

        output1 = colors.rgb_to_lab(output)
        target1 = colors.rgb_to_lab(target)
        # source = np.reshape(source, [-1, 3]).astype(np.float32)
        # target = np.reshape(target, [-1, 3]).astype(np.float32)
        # delta_e = np.sqrt(np.sum(np.power(source - target, 2), 1))
        output1 = torch.reshape(output1, [3, -1])
        target1 = torch.reshape(target1, [3, -1])

        delta_e = torch.sqrt(torch.sum(torch.pow(output1-target1,2),0))

        return torch.sum(delta_e)/(h*w)

    # \d??   9999  "" 9   [12]+  (12)+   12121212  1112212
    # (\d+?)(0*)  123000

class smooth_loss():
    @staticmethod
    def compute(att, x_kernel, y_kernel):
        #计算矩阵平滑损失函数
        smoothness_loss = (
            torch.sum(F.conv2d(att, x_kernel, stride=1) ** 2) +
            torch.sum(F.conv2d(att, y_kernel, stride=1) ** 2))
        return smoothness_loss