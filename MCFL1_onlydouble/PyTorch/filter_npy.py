import numpy as np
import os
import shutil

filePath='/home/csy/datasets/newSet1/'
fileList=os.listdir(filePath)
print(len(fileList))
matrix = np.load('train_all_5000_2.npy',allow_pickle=True).item()
npy_dic={}

for each in matrix.items():
    i = 0
    val=each[1]
    key=each[0]
    if key=='label_cc':
        break

    print(key)
    # print(val)
    val_list = val.tolist()
    print(len(val_list))
    new_val=[]
    for name in val_list:
        #print(name[26:])
        if name[26:] in fileList:
            new_val.append(name)
        else:
            #print(name)
            i+=1
    print(i)
    npy_dic[key] = new_val
npy_dic['label_cc']=matrix['label_cc']
npy_dic['name_list_cc']=matrix['name_list_cc']

#print(npy_dic)