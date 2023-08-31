########################################################################################################################
# 训练集验证集划分
########################################################################################################################

import os
import random

trainval_percent = 1
train_percent = 1

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")

    segfilepath = r'H:\BIGS\Video\0_code\Unet_Resnet\data\Masks/'
    saveBasePath = r'H:\BIGS\Video\0_code\Unet_Resnet\data\1/'

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("trainval size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    print("Generate txt in ImageSets done.")
