########################################################################################################################
# 数据增强脚本
########################################################################################################################
import os
from PIL import Image
import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class Rotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob

    def __call__(self, image, target):
        if random.random() < self.rotate_prob:
            value = [0, 90, 180, 270]
            r = random.sample(value, 1)
            image = F.rotate(image, angle=r[0])
            target = F.rotate(target, angle=r[0])

        return image, target


if __name__ == '__main__':
    path_in_img = r"D:\断层数据\PNG Patch/"                                                                          # 输入Images路径
    path_in_mask = r"D:\断层数据\Label_Mask3/"                                                                       # 输出Masks路径

    file_in_img = os.listdir(path_in_img)
    file_in_mask = os.listdir(path_in_mask)
    num_file_in = len(file_in_img)
    file_in_img.sort(key=lambda x: int(x[7:-5]))                                                                        # 控制顺序，按实际情况修改
    file_in_mask.sort(key=lambda x: int(x[7:-5]))

    m = 345                                                                                                               # 图像序号

    rotate_prob = 1                                                                                                     # 设置旋转概率
    flip_prob = 0.5                                                                                                     # 翻转概率

    randomFlip = RandomFlip(rotate_prob)
    rotate = Rotate(flip_prob)

    for j in range(0, 3):                                                                                               # 设置扩增的次数
        for i in range(0, num_file_in):

            m +=1

            img = Image.open(os.path.join(path_in_img, file_in_img[i]))
            target = Image.open(os.path.join(path_in_mask, file_in_mask[i]))

            if rotate_prob > 0:
                trans = [rotate]

            if flip_prob > 0:
                trans.append(randomFlip)

            compose = Compose(trans)
            transforms = compose

            img, target = transforms(img, target)

            img.save(r"D:\断层数据\aug_pic/{}_{}.png".format('aug', str(m)))                                              # 扩增Images保存路径
            target.save(r"D:\断层数据\aug_mask/{}_{}.png".format('aug', str(m)))                                           # 扩增Masks保存路径