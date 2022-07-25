"""
 Network structure of MCFL

 Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""

from .deep_wb_blocks import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize(tens,device):
    one = torch.ones(tens.shape, device=device)
    zero = torch.zeros(tens.shape, device=device)
    tens = torch.where(tens>= zero, tens, zero)
    tens = torch.where(tens <= one,tens, one)
    return tens

def showimg(img,device):
    # show预测图像
    #img = normalize(img, device)
    img = img[0].cpu().numpy().transpose((1, 2, 0))
    img = Image.fromarray((img * 255).astype(np.uint8))
    plt.imshow(img)
    plt.show()

class deepWBnet(nn.Module):
    def __init__(self,fre_sel_method1,fre_sel_method2,device):
        super(deepWBnet, self).__init__()   #super找deepWBnet的父类，nn.Module初始化
        self.n_channels = 3
        self.device=device

        ### FIPE
        #C*H*W->C*N*N
        #提取低频信息
        self.pre_img = FIPE(N=8,freq_sel_method=fre_sel_method1)

        ### out_block
        #高频+卷积后的低频
        self.out = SIG(in_channels=3,out_channels=3)

        ### CDCA
        self.att = CDCA(channel=384, freq_sel_method=fre_sel_method2)

        # self.att4 = SpatialAttention()
        #CDCA两边的网络结构
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24) ##卷积激活+卷积激活
        self.encoder_down1 = DownBlock(24, 48, B=False)#池化+卷积激活+卷积激活
        self.encoder_down2 = DownBlock(48, 96,B=False)#池化+卷积激活+卷积激活
        self.encoder_down3 = DownBlock(96, 192,B=False)#池化+卷积激活+卷积激活
        self.encoder_bridge_down = BridgeDown(192, 384,B=False) #池化+卷积激活

        self.decoder_bridge_up = BridgeUP(384, 192,B=False)#卷积激活+转置卷积
        self.decoder_up1 = UpBlock(192, 96,B=False)# 卷积激活+卷积激活+转置卷积
        self.decoder_up2 = UpBlock(96, 48,B=False)# 卷积激活+卷积激活+转置卷积
        self.decoder_up3 = UpBlock(48, 24,B=False)# 卷积激活+卷积激活+转置卷积
        self.decoder_out = OutputBlock(24, self.n_channels,B=False) # 卷积激活+卷积激活+卷积

        ### CDCA
        self.att_1 = CDCA(channel=384, freq_sel_method=fre_sel_method2)

        # self.att4 = SpatialAttention()
        # CDCA两边的网络结构
        self.encoder_inc_1 = DoubleConvBlock(self.n_channels, 24)  ##卷积激活+卷积激活
        self.encoder_down1_1 = DownBlock(24, 48, B=False)  # 池化+卷积激活+卷积激活
        self.encoder_down2_1 = DownBlock(48, 96, B=False)  # 池化+卷积激活+卷积激活
        self.encoder_down3_1 = DownBlock(96, 192, B=False)  # 池化+卷积激活+卷积激活
        self.encoder_bridge_down_1 = BridgeDown(192, 384, B=False)  # 池化+卷积激活

        self.decoder_bridge_up_1 = BridgeUP(384, 192, B=False)  # 卷积激活+转置卷积
        self.decoder_up1_1 = UpBlock(192, 96, B=False)  # 卷积激活+卷积激活+转置卷积
        self.decoder_up2_1 = UpBlock(96, 48, B=False)  # 卷积激活+卷积激活+转置卷积
        self.decoder_up3_1 = UpBlock(48, 24, B=False)  # 卷积激活+卷积激活+转置卷积
        self.decoder_out_1 = OutputBlock(24, self.n_channels, B=False)  # 卷积激活+卷积激活+卷积

    def output_matrix(self,x,filename):
        x_out=np.around(x[0].cpu().numpy(),decimals=4)
        with open('matrix/'+filename+'.txt','w') as f:
            f.write(str(x_out.shape)+'\n\n')
            for x_np in x_out:
                np.savetxt(f, x_np, fmt='%-7.4f')
                f.write('\n\n')
                np.savetxt(f, x_np, fmt='%-7.4f')
                f.write('\n\n')
                np.savetxt(f, x_np, fmt='%-7.4f')
                f.write('\n\n')

    def forward(self, x):
        #showimg(x,self.device)

        #self.output_matrix(x,'input')
        x_low, x_high = self.pre_img(x)
        # showimg(x_low, self.device)
        # showimg(x_high,self.device)

        ## x_low
        #self.output_matrix(x_low, 'low')

        x1 = self.encoder_inc(x_low)
        #self.output_matrix(x1,'encoder1')

        x2 = self.encoder_down1(x1)
        #self.output_matrix(x2,'encoder2')

        x3 = self.encoder_down2(x2)
        #self.output_matrix(x3, 'encoder3')

        x4 = self.encoder_down3(x3)
        #self.output_matrix(x4,'encoder4')

        x5 = self.encoder_bridge_down(x4)
        #self.output_matrix(x5, 'encoder5')

        ### att
        x5 = self.att(x5) * x5
        #self.output_matrix(x5, 'att')

        x6 = self.decoder_bridge_up(x5)
        #self.output_matrix(x, 'decoder1')

        x7 = self.decoder_up1(x6, x4)
        #self.output_matrix(x, 'decoder2')

        x8 = self.decoder_up2(x7, x3)
        #self.output_matrix(x, 'decoder3')

        x9 = self.decoder_up3(x8, x2)
        #self.output_matrix(x, 'decoder4')

        x_low_out = self.decoder_out(x9, x1)
        # self.output_matrix(x_low_out, 'decoder5')

        #showimg(x_low_out,self.device)

        x1 = self.encoder_inc_1(x_low_out)
        # self.output_matrix(x1,'encoder1')

        x2 = self.encoder_down1_1(x1)
        # self.output_matrix(x2,'encoder2')

        x3 = self.encoder_down2_1(x2)
        # self.output_matrix(x3, 'encoder3')

        x4 = self.encoder_down3_1(x3)
        # self.output_matrix(x4,'encoder4')

        x5 = self.encoder_bridge_down_1(x4)
        # self.output_matrix(x5, 'encoder5')

        ### att
        x5 = self.att_1(x5) * x5
        # self.output_matrix(x5, 'att')

        x6 = self.decoder_bridge_up_1(x5)
        # self.output_matrix(x, 'decoder1')

        x7 = self.decoder_up1_1(x6, x4)
        # self.output_matrix(x, 'decoder2')

        x8 = self.decoder_up2_1(x7, x3)
        # self.output_matrix(x, 'decoder3')

        x9 = self.decoder_up3_1(x8, x2)
        # self.output_matrix(x, 'decoder4')

        x_low_out = self.decoder_out_1(x9, x1)
        # self.output_matrix(x_low_out, 'decoder5')

        #showimg(x_low_out, self.device)

        out_c = self.out(x_high,x_low_out)
        #self.output_matrix(out_c, 'output')


        out_c = normalize(out_c, self.device)
        #showimg(out_c, self.device)

        return out_c
