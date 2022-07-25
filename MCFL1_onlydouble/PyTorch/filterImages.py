import os
import shutil

filePath='/home/csy/datasets/Set1/'
newPath='/home/csy/datasets/newSet1/'
fileList=os.listdir(filePath)
baseNames=[]
gt_ext = ('G_AS.png', 'T_AS.jpg', 'S_AS.jpg')
for img in fileList:
    parts = img.split('_')
    base_name = ''
    for i in range(len(parts) - 2):
        base_name = base_name + parts[i] + '_'
    baseNames.append(base_name)
i=0
for name in baseNames:
    gt_awb_file = name + gt_ext[0]
    #print(gt_awb_file)
    gt_t_file = name + gt_ext[1]
    gt_s_file = name + gt_ext[2]
    if (gt_awb_file in fileList) and (gt_t_file in fileList) and (gt_s_file in fileList):
        print(i)
        #print('√')
        for img in fileList:
             if img.find(name)!=-1:
                shutil.copyfile(filePath+img, newPath+img)  # 复制文件
    else:
        print(name)
    i=i+1

    # gt_ext = ('G_AS.png', 'T_AS.jpg', 'S_AS.jpg')
    # # self.imgfiles.remove('/home/csy/datasets/Set1/NikonD5200_0178')
    # for img in self.imgfiles:
    #     parts = img.split('_')
    #     base_name = ''
    #     for i in range(len(parts) - 2):
    #         base_name = base_name + parts[i] + '_'
    #     # print(base_name)
    #
    #     gt_awb_file = base_name + gt_ext[0]
    #     if not os.path.exists(gt_awb_file):
    #         msg = gt_awb_file + " does not exist."
    #         print(msg)
    #         self.imgfiles.remove(img)
    #         continue
    #
    #     gt_t_file = base_name + gt_ext[1]
    #     if not os.path.exists(gt_t_file):
    #         msg = gt_t_file + " does not exist."
    #         print(msg)
    #         self.imgfiles.remove(img)
    #         continue
    #
    #     gt_s_file = base_name + gt_ext[2]
    #     if not os.path.exists(gt_s_file):
    #         msg = gt_s_file + " does not exist."
    #         print(msg)
    #         self.imgfiles.remove(img)
    #         continue