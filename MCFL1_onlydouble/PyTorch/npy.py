# import numpy as np

import numpy as np
import h5py
from PIL import Image
matrix = np.load('test_set2_all_.npy',allow_pickle=True).item()
npy_dic=matrix
#print(matrix)
#print(matrix.keys())
for each in matrix.items():
    val=each[1]
    key=each[0]
    if key=='label':
        break
    # print('old')
    # print(val)
    val_list = val.tolist()
    #print(type(val_list))
    new_list=[]
    for v in val_list:
         # v=str(v)
        v=v.replace('/dataset/lcx/set2/Set2_input_images/','/home/csy/datasets/Set2/')
        v = v.replace('/dataset/lcx/set2/Set2_gt_images/', '/home/csy/datasets/Set2_gt/')
        try:
             img = Image.open(v)
        except FileNotFoundError:
             print(v)
             continue
        # v=v.replace('png','jpg')
        # v=v.replace('G_AS.jpg','G_AS.png')
        new_list.append(v)
    new_val=np.array(new_list)
    npy_dic[key]=new_val
    # print('new')
    #print(new_val)
print(npy_dic)
np.save('test_set2_all_1.npy',npy_dic)


# f = h5py.File('train.mat', 'w')
# f.create_dataset('data', data=matrix)
# 这里不会将数据转置