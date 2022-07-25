import torch.nn as nn
import torch
import torch.nn.functional as F
from utilities.loss_func import mae_loss,smooth_loss

def get_sobel_kernel(device, chnls=1):
  #生成xy方向的核，用于计算矩阵平滑性
  x_kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
  x_kernel = torch.tensor(x_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)   #unsqueeze增加维度,expand维度复制扩展
  x_kernel.requires_grad = False
  y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  y_kernel = torch.tensor(y_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)
  y_kernel.requires_grad = False
  return x_kernel, y_kernel

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        #两个Loss参数，加入到模型优化的Parameter列表中，初始化为0
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)), requires_grad=True)

    def forward(self, input, targets, label):
        if self.task_num == 2:
            out, out_p = self.model(input)  #输出图像、映射矩阵

            precision1 = torch.exp(-self.log_vars[0])   #参数一
            loss1 = mae_loss.compute(out, targets)  #输出图像与输入图像之间的绝对值误差
            loss1_w = torch.sum(precision1 * loss1 + self.log_vars[0], -1)  #乘上参数一

            precision2 = torch.exp(-self.log_vars[1])   #参数二
            #计算矩阵xy方向的梯度损失函数
            x_kernel, y_kernel = get_sobel_kernel(self.device, chnls=3)
            loss2 = smooth_loss.compute(out_p, x_kernel,y_kernel)
            loss2_w = torch.sum(precision2 * loss2 + self.log_vars[1], -1)  #乘上参数二

            loss = loss1_w + loss2_w    #总损失

            return out, out_p, loss, loss1, loss2, self.log_vars.data.tolist()

        elif self.task_num == 1:
            out = self.model(input)
            loss1 = mae_loss.compute(out, targets)
            return out, loss1

        elif self.task_num == 3:  ### 场景
            out, out_p = self.model(input)

            precision1 = torch.exp(-self.log_vars[0])
            loss1 = mae_loss.compute(out, targets)
            loss1_w = torch.sum(precision1 * loss1 + self.log_vars[0], -1)

            precision21 = torch.exp(-self.log_vars[1])
            precision22 = torch.exp(-self.log_vars[2])
            loss21, loss22 = CCLoss_new.compute(out_p, label)
            loss2_w = torch.sum(precision21 * loss21 + self.log_vars[1], -1) + torch.sum(
                precision22 * loss22 + self.log_vars[2], -1)

            loss = loss1_w + loss2_w

            return out, out_p, loss, loss1, loss21, loss22, self.log_vars.data.tolist()