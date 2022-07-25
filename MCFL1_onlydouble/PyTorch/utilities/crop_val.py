import torch
import math
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torchvision import transforms
import matplotlib.pyplot as plt

class MultiEvalModule(DataParallel):
    def __init__(self, model, base_size, crop_size, device_ids=None,
                 multi_scales=False):
        super().__init__(model)
        self.base_size = base_size
        self.crop_size = crop_size
        self.model = model
        if not multi_scales:
            self.scales = [1.0]
        else:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]

        # print('MultiEvalModule: base_size {}, crop_size {}'. \
        #       format(self.base_size, self.crop_size))


    def forward(self,image):
        # 多网格裁剪
        batch, c, h, w = image.size()
        self.base_size = max(h, w)
        assert (batch == 1)
        if len(self.scales) == 1:  # single scale
            stride_rate = 2.0 / 3.0
        else:
            stride_rate = 1.0 / 2.0
        crop_size = self.crop_size

        stride = int(crop_size * stride_rate)


        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, c, h, w).zero_().cuda()


        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))

            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = self.resize_image(image, height, width)
            ### 图像尺寸小于要裁剪的尺寸
            if long_size <= crop_size:
                pad_img = self.pad_image(cur_img, crop_size)
                outputs = self.model(pad_img)
                outputs = self.crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = self.pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = self.crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = self.pad_image(crop_img, crop_size)
                        output = self.model(pad_crop_img)
                        outputs[:, :, h0:h1, w0:w1] += self.crop_image(output,
                                                                  0, h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            #### output可视化
            # tf = transforms.ToPILImage()
            #
            # input = image.cpu().clone()
            # input = input[0]
            # input_img = tf(input)
            # plt.imshow(input_img)
            # plt.show()
            # # name = name_dir + 'img.png'
            # out = outputs.cpu().clone()
            # out = out[0]
            # out_img = tf(out)
            # plt.imshow(out_img)
            # plt.show()

            score = self.resize_image(outputs, h, w)
            scores += score

        return scores

    def resize_image(self,img, h, w):
        return F.interpolate(img, (h, w))

    def pad_image(self,img, crop_size):
        b, c, h, w = img.size()
        assert (c == 3)
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        # pad_values = np.mean(img[:,:,-padh,:-padh],axis=(2,3))
        # pad_values = -np.array(mean) / np.array(std)
        img_pad = img.new().resize_(b, c, h + padh, w + padw)
        for i in range(c):
            # note that pytorch pad params is in reversed orders
            # valu = float(np.mean(img[:, i, :-padw, :-padh])[0])
            img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=0)
        assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
        return img_pad

    def crop_image(self,img, h0, h1, w0, w1):
        return img[:, :, h0:h1, w0:w1]

    def module_inference(self,model, image):
        output = model(image)
        return output.exp()


class MultiEvalModule_l(DataParallel):
    def __init__(self, model, base_size, crop_size, device_ids=None,
                 multi_scales=False):
        super().__init__(model)
        self.base_size = base_size
        self.crop_size = crop_size
        self.model = model
        if not multi_scales:
            self.scales = [1.0]
        else:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]

        # print('MultiEvalModule: base_size {}, crop_size {}'. \
        #       format(self.base_size, self.crop_size))


    def forward(self,image):
        # 多网格裁剪
        batch, c, h, w = image.size()
        self.base_size = max(h, w)
        assert (batch == 1)
        if len(self.scales) == 1:  # single scale
            stride_rate = 2.0 / 3.0
        else:
            stride_rate = 1.0 / 2.0
        crop_size = self.crop_size

        stride = int(crop_size * stride_rate)


        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, c, h, w).zero_().cuda()


        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))

            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = self.resize_image(image, height, width)
            ### 图像尺寸小于要裁剪的尺寸
            if long_size <= crop_size:
                pad_img = self.pad_image(cur_img, crop_size)
                outputs,_,_,_,_ = self.model(pad_img)
                outputs = self.crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = self.pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = self.crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = self.pad_image(crop_img, crop_size)
                        output,_,_,_,_ = self.model(pad_crop_img)
                        outputs[:, :, h0:h1, w0:w1] += self.crop_image(output,
                                                                  0, h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            #### output可视化
            # tf = transforms.ToPILImage()
            #
            # input = image.cpu().clone()
            # input = input[0]
            # input_img = tf(input)
            # plt.imshow(input_img)
            # plt.show()
            # # name = name_dir + 'img.png'
            # out = outputs.cpu().clone()
            # out = out[0]
            # out_img = tf(out)
            # plt.imshow(out_img)
            # plt.show()

            score = self.resize_image(outputs, h, w)
            scores += score

        return scores

    def resize_image(self,img, h, w):
        return F.interpolate(img, (h, w))

    def pad_image(self,img, crop_size):
        b, c, h, w = img.size()
        assert (c == 3)
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        # pad_values = np.mean(img[:,:,-padh,:-padh],axis=(2,3))
        # pad_values = -np.array(mean) / np.array(std)
        img_pad = img.new().resize_(b, c, h + padh, w + padw)
        for i in range(c):
            # note that pytorch pad params is in reversed orders
            # valu = float(np.mean(img[:, i, :-padw, :-padh])[0])
            img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=0)
        assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
        return img_pad

    def crop_image(self,img, h0, h1, w0, w1):
        return img[:, :, h0:h1, w0:w1]

    def module_inference(self,model, image):
        output,_,_,_,_ = model(image)
        return output.exp()


class MultiEvalModule_ll(DataParallel):
    def __init__(self, model, base_size, crop_size, device_ids=None,
                 multi_scales=False):
        super().__init__(model)
        self.base_size = base_size
        self.crop_size = crop_size
        self.model = model
        if not multi_scales:
            self.scales = [1.0]
        else:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]

        # print('MultiEvalModule: base_size {}, crop_size {}'. \
        #       format(self.base_size, self.crop_size))


    def forward(self,image):
        # 多网格裁剪
        batch, c, h, w = image.size()
        self.base_size = max(h, w)
        assert (batch == 1)
        if len(self.scales) == 1:  # single scale
            stride_rate = 2.0 / 3.0
        else:
            stride_rate = 1.0 / 2.0
        crop_size = self.crop_size

        stride = int(crop_size * stride_rate)


        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, c, h, w).zero_().cuda()


        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))

            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = self.resize_image(image, height, width)
            ### 图像尺寸小于要裁剪的尺寸
            if long_size <= crop_size:
                pad_img = self.pad_image(cur_img, crop_size)
                outputs,_ = self.model(pad_img)
                outputs = self.crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = self.pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = self.crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = self.pad_image(crop_img, crop_size)
                        output,_ = self.model(pad_crop_img)
                        outputs[:, :, h0:h1, w0:w1] += self.crop_image(output,
                                                                  0, h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            #### output可视化
            # tf = transforms.ToPILImage()
            #
            # input = image.cpu().clone()
            # input = input[0]
            # input_img = tf(input)
            # plt.imshow(input_img)
            # plt.show()
            # # name = name_dir + 'img.png'
            # out = outputs.cpu().clone()
            # out = out[0]
            # out_img = tf(out)
            # plt.imshow(out_img)
            # plt.show()

            score = self.resize_image(outputs, h, w)
            scores += score

        return scores

    def resize_image(self,img, h, w):
        return F.interpolate(img, (h, w))

    def pad_image(self,img, crop_size):
        b, c, h, w = img.size()
        assert (c == 3)
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        # pad_values = np.mean(img[:,:,-padh,:-padh],axis=(2,3))
        # pad_values = -np.array(mean) / np.array(std)
        img_pad = img.new().resize_(b, c, h + padh, w + padw)
        for i in range(c):
            # note that pytorch pad params is in reversed orders
            # valu = float(np.mean(img[:, i, :-padw, :-padh])[0])
            img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=0)
        assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
        return img_pad

    def crop_image(self,img, h0, h1, w0, w1):
        return img[:, :, h0:h1, w0:w1]

    def module_inference(self,model, image):
        output,_ = model(image)
        return output.exp()


class MultiEvalModule_lll(DataParallel):
    def __init__(self, model, base_size, crop_size, device_ids=None,
                 multi_scales=False,stage='2'):
        super().__init__(model)
        self.base_size = base_size
        self.crop_size = crop_size
        self.model = model
        self.stage = stage
        if not multi_scales:
            self.scales = [1.0]
        else:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]

        # print('MultiEvalModule: base_size {}, crop_size {}'. \
        #       format(self.base_size, self.crop_size))


    def forward(self,image):
        # 多网格裁剪
        batch, c, h, w = image.size()
        self.base_size = max(h, w)
        assert (batch == 1)
        if len(self.scales) == 1:  # single scale
            stride_rate = 2.0 / 3.0
        else:
            stride_rate = 1.0 / 2.0
        crop_size = self.crop_size

        stride = int(crop_size * stride_rate)


        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, c, h, w).zero_().cuda()


        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))

            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = self.resize_image(image, height, width)
            ### 图像尺寸小于要裁剪的尺寸
            if long_size <= crop_size:
                pad_img = self.pad_image(cur_img, crop_size)
                if self.stage == '1':
                    output,_,_ = self.model(pad_img)
                    outputs = self.crop_image(output, 0, height, 0, width)
                elif self.stage == '2':
                    #_, output, _ = self.model(pad_img)
                    output= self.model(pad_img)
                    outputs = self.crop_image(output, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = self.pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 3, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = self.crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = self.pad_image(crop_img, crop_size)
                        if self.stage == '1':
                            output,_,_ = self.model(pad_crop_img)
                            outputs[:, :, h0:h1, w0:w1] += self.crop_image(output,
                                                                  0, h1 - h0, 0, w1 - w0)

                        elif self.stage == '2':
                            _, output, _ = self.model(pad_crop_img)
                            outputs[:, :, h0:h1, w0:w1] += self.crop_image(output,
                                                                           0, h1 - h0, 0, w1 - w0)

                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            #### output可视化
            # tf = transforms.ToPILImage()
            #
            # input = image.cpu().clone()
            # input = input[0]
            # input_img = tf(input)
            # plt.imshow(input_img)
            # plt.show()
            # # name = name_dir + 'img.png'
            # out = outputs.cpu().clone()
            # out = out[0]
            # out_img = tf(out)
            # plt.imshow(out_img)
            # plt.show()

            score = self.resize_image(outputs, h, w)
            scores += score

        return scores

    def resize_image(self,img, h, w):
        return F.interpolate(img, (h, w))

    def pad_image(self,img, crop_size):
        b, c, h, w = img.size()
        assert (c == 3)
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        # pad_values = np.mean(img[:,:,-padh,:-padh],axis=(2,3))
        # pad_values = -np.array(mean) / np.array(std)
        img_pad = img.new().resize_(b, c, h + padh, w + padw)
        for i in range(c):
            # note that pytorch pad params is in reversed orders
            # valu = float(np.mean(img[:, i, :-padw, :-padh])[0])
            img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=0)
        assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
        return img_pad

    def crop_image(self,img, h0, h1, w0, w1):
        return img[:, :, h0:h1, w0:w1]

    def module_inference(self,model, image):
        output1,output2,_ = model(image)
        return output1.exp(),output2.exp()
