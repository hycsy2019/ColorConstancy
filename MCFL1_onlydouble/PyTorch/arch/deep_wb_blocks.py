"""
 The main blocks of MCFL

 Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""

import torch.nn as nn

import torch.nn.functional as F
from .zigzag import *
import math

class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels, B=False):
        super().__init__()
        if B:   #卷积+激活+归一+卷积+激活+归一
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),   #进行归一化防止数据过大而导致网络性能不稳定
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        else:   #卷积+激活+卷积+激活
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)




class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels,B=B)
        )   #池化+卷积激活+卷积激活

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        if B:   #池化+卷积+激活+归一
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        else:   #池化+卷积+激活
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        if B:   #卷积激活+归一+转置卷积
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

        else:   #卷积激活+转置卷积
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

    def forward(self, x):
        return self.conv_up(x)


class BridgeUP1(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,output_padding=1)
        )

    def forward(self, x):
        return self.conv_up(x)


class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        # 卷积激活+卷积激活+转置卷积
        self.conv = DoubleConvBlock(in_channels * 2, in_channels,B)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.out_conv = nn.Sequential(
            #### 尝试加入
            # 卷积激活+卷积激活+卷积
            DoubleConvBlock(in_channels * 2, in_channels,B),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)


class upsample(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self,):
        super().__init__()

        # squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.up = nn.Sequential(
            Fire(512,32,256,256),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),
            Fire(256, 32, 128, 128),  ## output 256
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            Fire(128, 16, 64, 64),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            Fire(64, 8, 32, 32),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            Fire(32, 8, 16, 16),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16,3,kernel_size=1)
        )


    def forward(self, x):
        return self.up(x)


def get_freq_indices(method):
    # assert method is 'low'
    # num_freq = idx
    # num_freq = int(method[3:])
    if 'low' in method:  ## x 20. y 20
        num_freq = int(method[3:])  #取low后的数字
        #z字形坐标索引
        all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5] #长度21
        all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        mapper_x = all_low_indices_x[:num_freq+1]
        mapper_y = all_low_indices_y[:num_freq+1]
    elif '0' in method:
        mapper_x =[0]
        mapper_y =[0]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class SIG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIG, self).__init__()

        #卷积+激活+卷积+激活+卷积
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels*2),
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1))

    def forward(self,x1,x2):
        x = x1+x2
        return self.out_conv(x)


class FIPE(nn.Module):
    def __init__(self, N, freq_sel_method='low0'):
        super(FIPE, self).__init__()
        self.N = N

        #### mask
        self.register_buffer('weight', self.init_cos()) #开辟一个1*1*self.N*self.N的缓存大小空间
        self.method = freq_sel_method

        # self.weight = self.init_cos()
        mapper_x,mapper_y = get_freq_indices(self.method)
        self.mapper_x = [temp_x * (self.N // 8) for temp_x in mapper_x]#//整数除法，向下取整
        self.mapper_y = [temp_y * (self.N // 8) for temp_y in mapper_y]
        # print('d')

    def forward(self,x):
        ### img [b,c,h,w]
        n, c, h, w = x.shape

        x_1 = torch.split(x, self.N, dim=2)  # [b,c,h,w] → h/8 * [b,c,8,w]
        x_2 = torch.cat(x_1, dim=1)  # → [b, c * h/8, 8, w]
        x_3 = torch.split(x_2, self.N, dim=3)  # [b, c * h/8, 8, w] → w/8 * [b, c * h/8, 8, 8]
        x_4 = torch.cat(x_3, dim=1)  # [b, c []* h/8 * w/8 , 8, 8]


        fre = torch.einsum('pijk,qtkl->qtjl', self.weight, x_4)  # (256,8,8)
        fre = torch.einsum('pijk,qtkl->pijl', fre, self.weight.permute(0, 1, 3, 2)) # (256,8,8)

        ### index
        fre_low = self.low_fre(self.mapper_x, self.mapper_y, fre)

        x_low = torch.einsum('pijk,qtkl->qtjl', self.weight.permute(0, 1, 3, 2), fre_low)
        x_low = torch.einsum('pijk,qtkl->pijl', x_low, self.weight)
        n_low, c_low, h_low, w_low = x_low.shape
        x_low = torch.split(x_low, int(c_low/(w/self.N)), dim=1)
        x_low = torch.cat(x_low, dim=3)
        x_low = torch.split(x_low, c, dim=1)
        x_low = torch.cat(x_low, dim=2)

        ### x_high
        x_high = x-x_low

        return x_low, x_high



    def low_fre(self,mapper_x, mapper_y,fre):
        n, c, h, w = fre.shape
        low = torch.zeros_like(fre)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            low[:,:,u_x,v_y] = fre[:,:,u_x,v_y]
        # print(low)
        return low


    def init_cos(self):
        A = torch.zeros((1, 1, self.N, self.N))     #返回张量
        A[0, 0, 0, :] = 1 * math.sqrt(1 / self.N)
        for i in range(1, self.N):
            for j in range(self.N):
                A[0, 0, i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * self.N)
                                    ) * math.sqrt(2 / self.N)
        return A



class CDCA(torch.nn.Module):
    def __init__(self, channel, dct_h=8, dct_w=8, reduction=16, freq_sel_method='low2'):
    #reduction：相机差异减少
        super(CDCA, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = self.get_freq_indices1(freq_sel_method)

        self.num_split = 1

        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]
        #CDCA变换层
        self.dct_layer = CDCA_layer(dct_h, dct_w, mapper_x, mapper_y, channel)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel, 1, bias=False)


        self.sigmoid = nn.Sigmoid()


        #池化层，输出1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        n, c, h, w = x.shape
        # x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))#二元自适应均值汇聚层
        else:
            x_pooled = x


        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fre_out = self.fc2(self.relu1(self.fc1(self.dct_layer(x_pooled))))

        w = avg_out + max_out + fre_out

        w = self.sigmoid(w)
        return w

    def get_freq_indices1(self,method):
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5,6,5,4,3,2,1,0,0,1,2]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0,0,1,2,3,4,5,6,7,6,5]
            mapper_x = [all_low_indices_x[num_freq]]
            mapper_y = [all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y


class CDCA_m(torch.nn.Module):
    def __init__(self, channel, dct_h=8, dct_w=8, reduction=16, freq_sel_method1='low20',freq_sel_method2='low18',freq_sel_method3='low6',freq_sel_method4='low7',freq_sel_method5='low8'):
        super(CDCA_m, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        self.dct_layer1 = self.dctlayer(freq_sel_method1, dct_h, dct_w, channel)
        self.dct_layer2 = self.dctlayer(freq_sel_method2, dct_h, dct_w, channel)
        if not (freq_sel_method3 == None) and not (freq_sel_method4 == None) and not (freq_sel_method5 == None):
            self.dct_layer3 = self.dctlayer(freq_sel_method3, dct_h, dct_w, channel)
            self.dct_layer4 = self.dctlayer(freq_sel_method4, dct_h, dct_w, channel)
            self.dct_layer5 = self.dctlayer(freq_sel_method5, dct_h, dct_w, channel)
            self.E = True
        else:
            self.E = False

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def dctlayer(self,freq_sel_method1,dct_h,dct_w,channel):
        mapper_x1, mapper_y1 = self.get_freq_indices1(freq_sel_method1)

        mapper_x1 = [temp_x * (dct_h // 8) for temp_x in mapper_x1]
        mapper_y1 = [temp_y * (dct_w // 8) for temp_y in mapper_y1]

        dct_layer1 = CDCA_layer(dct_h, dct_w, mapper_x1, mapper_y1, channel)
        return dct_layer1

    def forward(self, x):
        n, c, h, w = x.shape
        # x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pooled = x

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fre_out1 = self.fc2(self.relu1(self.fc1(self.dct_layer1(x_pooled))))
        fre_out2 = self.fc2(self.relu1(self.fc1(self.dct_layer2(x_pooled))))
        if self.E:
            fre_out3 = self.fc2(self.relu1(self.fc1(self.dct_layer3(x_pooled))))
            fre_out4 = self.fc2(self.relu1(self.fc1(self.dct_layer4(x_pooled))))
            fre_out5 = self.fc2(self.relu1(self.fc1(self.dct_layer5(x_pooled))))
            w = avg_out + max_out + fre_out1 + fre_out2 + fre_out3 + fre_out4 + fre_out5
        else:
            w = avg_out + max_out + fre_out1 + fre_out2

        # w = avg_out + fre_out
        # w = avg_out + fre_out1 + fre_out2 + fre_out3
        w = self.sigmoid(w)
        return w

    def get_freq_indices1(self,method):
        if 'low' in method:  ## x 20. y 20
            num_freq = int(method[3:])
            all_low_indices_x = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5]
            all_low_indices_y = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
            mapper_x = [all_low_indices_x[num_freq]]
            mapper_y = [all_low_indices_y[num_freq]]
        elif '0' in method:
            mapper_x = [0]
            mapper_y = [0]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y


class CDCA_layer(nn.Module):

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(CDCA_layer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        #开辟转换矩阵的缓存空间
        self.register_buffer('weight', self.get_dct_filter1(height, width, mapper_x, mapper_y, channel))



    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        # x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        result = torch.unsqueeze(result,dim=2)
        result = torch.unsqueeze(result, dim=3)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  #转换矩阵A(freq,pos)位置处的值
        if freq == 0:   #i=0,c(i)=sqrt(1/N)
            return result
        else:   #i!=0,c(i)=sqrt(2/N)
            return result * math.sqrt(2)

    def get_dct_filter1(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)  #生成dct转换矩阵
                    # print('done')

        return dct_filter