"""
 Dataloader
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

from os.path import join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageOps
import random
from scipy.io import loadmat
import os.path

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, fold=0, patch_size=128, patch_num_per_image=1, max_trdata=12000):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_image = patch_num_per_image
        # get selected training data based on the current fold
        if fold is not 0:   #验证集为fold
            tfolds = list(set([1, 2, 3]) - set([fold])) #排除验证集
            logging.info(f'Training process will use {max_trdata} training images randomly selected from folds {tfolds}')
            files = loadmat(join('..', 'folds', 'fold%d_.mat' % fold))  #读取训练.mat数据，返回词典
            files = files['training']   #筛出训练图像
            self.imgfiles = []
            logging.info('Loading training images information...')
            for i in range(len(files)):
                temp_files = glob(imgs_dir + files[i][0][0])    #返回所有匹配的文件路径列表
                for file in temp_files:
                    self.imgfiles.append(file)
        elif fold is 0: #无交叉验证
            logging.info(f'Training process will use {max_trdata} training images randomly selected from all training data')
            logging.info('Loading training images information...')
            # npy_data = np.load('test_set2_all_1.npy', allow_pickle=True).item()
            # self.imgfiles = npy_data['input'].tolist()
            #print(self.imgfiles)
            self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                        if not file.startswith('.')]
        else:
            logging.info(f'There is no fold {fold}! Training process will use all training data.')

        if max_trdata is not 0 and len(self.imgfiles) > max_trdata: #文件数大于设置的训练图像数
            random.shuffle(self.imgfiles)
            self.imgfiles = self.imgfiles[0:max_trdata]
        logging.info(f'Creating dataset with {len(self.imgfiles)} examples')
        # self.clearImgFiles()
        # print("data length:"+str(len(self.imgfiles)))

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op is 1:
            pil_img = ImageOps.mirror(pil_img)
        elif flip_op is 2:
            pil_img = ImageOps.flip(pil_img)

        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        #set2
        # img_file = self.imgfiles[i]
        # in_img = Image.open(img_file)
        # # get image size
        # w, h = in_img.size
        # # get ground truth images
        # gt_awb_file =img_file.replace('/home/csy/datasets/Set2/','/home/csy/datasets/Set2_gt/')
        # try:
        #     awb_img = Image.open(gt_awb_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_awb_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # # get flipping option
        # flip_op = np.random.randint(3)
        # # get random patch coord
        # patch_x = np.random.randint(0, high=w - self.patch_size)
        # patch_y = np.random.randint(0, high=h - self.patch_size)
        #
        # in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        # awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        # for j in range(self.patch_num_per_image - 1):
        #     # get flipping option
        #     flip_op = np.random.randint(3)
        #     # get random patch coord
        #     patch_x = np.random.randint(0, high=w - self.patch_size)
        #     patch_y = np.random.randint(0, high=h - self.patch_size)
        #     temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     in_img_patches = np.append(in_img_patches, temp, axis=0)
        #     temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     awb_img_patches = np.append(awb_img_patches, temp, axis=0)
        # return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches)}

        #set1
        # gt_ext = ('G_AS.png', 'T_AS.jpg', 'S_AS.jpg')
        # img_file = self.imgfiles[i]
        # in_img = Image.open(img_file)
        # # get image size
        # w, h = in_img.size
        # # get ground truth images
        # parts = img_file.split('_')
        # base_name = ''
        # for i in range(len(parts) - 2):
        #     base_name = base_name + parts[i] + '_'
        # # print(base_name)
        #
        # gt_awb_file = base_name + gt_ext[0]
        # try:
        #     awb_img = Image.open(gt_awb_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_awb_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # gt_t_file = base_name + gt_ext[1]
        # try:
        #     t_img = Image.open(gt_t_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_t_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # gt_s_file = base_name + gt_ext[2]
        # try:
        #     s_img = Image.open(gt_s_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_s_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # # get flipping option
        # flip_op = np.random.randint(3)
        # # get random patch coord
        # patch_x = np.random.randint(0, high=w - self.patch_size)
        # patch_y = np.random.randint(0, high=h - self.patch_size)
        #
        # in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        # awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        # img_t_patches = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
        # img_s_patches = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
        # for j in range(self.patch_num_per_image - 1):
        #     # get flipping option
        #     flip_op = np.random.randint(3)
        #     # get random patch coord
        #     patch_x = np.random.randint(0, high=w - self.patch_size)
        #     patch_y = np.random.randint(0, high=h - self.patch_size)
        #     temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     in_img_patches = np.append(in_img_patches, temp, axis=0)
        #     temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     awb_img_patches = np.append(awb_img_patches, temp, axis=0)
        #     temp = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     img_t_patches = np.append(img_t_patches, temp, axis=0)
        #     temp = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     img_s_patches = np.append(img_s_patches, temp, axis=0)
        # return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches),
        #         'gt-T': torch.from_numpy(img_t_patches), 'gt-S': torch.from_numpy(img_s_patches)}

        #Cube+
        # gt_ext = ('AS.JPG', 'T.JPG', 'S.JPG')
        # img_file = self.imgfiles[i]
        # in_img = Image.open(img_file)
        # # get image size
        # w, h = in_img.size
        # # get ground truth images
        # parts = img_file.split('_')
        # base_name = ''
        # for i in range(len(parts) - 1):
        #     base_name = base_name + parts[i] + '_'
        # # print(base_name)
        #
        # gt_awb_file = base_name + gt_ext[0]
        # try:
        #     awb_img = Image.open(gt_awb_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_awb_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # gt_t_file = base_name + gt_ext[1]
        # try:
        #     t_img = Image.open(gt_t_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_t_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # gt_s_file = base_name + gt_ext[2]
        # try:
        #     s_img = Image.open(gt_s_file)
        # except FileNotFoundError:
        #     msg = "Sorry, the file " + gt_s_file + " does not exist."
        #     print(msg)
        #     return {}
        #
        # # get flipping option
        # flip_op = np.random.randint(3)
        # # get random patch coord
        # patch_x = np.random.randint(0, high=w - self.patch_size)
        # patch_y = np.random.randint(0, high=h - self.patch_size)
        #
        # in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        # awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        # img_t_patches = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
        # img_s_patches = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
        # for j in range(self.patch_num_per_image - 1):
        #     # get flipping option
        #     flip_op = np.random.randint(3)
        #     # get random patch coord
        #     patch_x = np.random.randint(0, high=w - self.patch_size)
        #     patch_y = np.random.randint(0, high=h - self.patch_size)
        #     temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     in_img_patches = np.append(in_img_patches, temp, axis=0)
        #     temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     awb_img_patches = np.append(awb_img_patches, temp, axis=0)
        #     temp = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     img_t_patches = np.append(img_t_patches, temp, axis=0)
        #     temp = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
        #     img_s_patches = np.append(img_s_patches, temp, axis=0)
        # return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches),
        #         'gt-T': torch.from_numpy(img_t_patches), 'gt-S': torch.from_numpy(img_s_patches)}

        # Mixed
        gt_ext = 'G_AS.jpg'
        img_file = self.imgfiles[i]
        in_img = Image.open(img_file)
        # get image size
        w, h = in_img.size
        # get ground truth images
        parts = img_file.split('_')
        base_name = ''
        for i in range(len(parts) - 2):
            base_name = base_name + parts[i] + '_'
        # print(base_name)

        gt_awb_file = base_name + gt_ext
        try:
            awb_img = Image.open(gt_awb_file)
        except FileNotFoundError:
            msg = "Sorry, the file " + gt_awb_file + " does not exist."
            print(msg)
            return {}

        # get flipping option
        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)

        in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)

        for j in range(self.patch_num_per_image - 1):
            # get flipping option
            flip_op = np.random.randint(3)
            # get random patch coord
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)
            temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
            in_img_patches = np.append(in_img_patches, temp, axis=0)
            temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
            awb_img_patches = np.append(awb_img_patches, temp, axis=0)

        return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches)}
