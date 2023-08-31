########################################################################################################################
# 图像裁剪脚本
########################################################################################################################
import time
import os
import numpy as np
from osgeo import gdal

class GRID:

    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):

        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image


if __name__ == '__main__':

    path_img = r"H:\data\test\0_测试影像/test.tif"                                                                        # 输入图像路径
    path_out = r"H:\data\test\1/"                                                                                       # 输出图像文件夹路径

    t_start = time.time()

    run = GRID()
    proj, geotrans, data = run.load_image(path_img)

    channel, height, width = data.shape

    patch_size_w = 256
    patch_size_h = 256

    num = 0                                                                                                             # 图像名字序号


    for i in range(height // patch_size_h):
        for j in range(width // patch_size_w):
            num += 1

            sub_image = data[:, i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w]


            px = geotrans[0] + j * patch_size_w * geotrans[1] + i * patch_size_h * geotrans[2]
            py = geotrans[3] + j * patch_size_w * geotrans[4] + i * patch_size_h * geotrans[5]
            new_geotrans = [px, geotrans[1], geotrans[2], py, geotrans[4], geotrans[5]]

            run.write_image(path_out + '{}.tif'.format(num), proj, new_geotrans, sub_image)
            time_end = time.time()
            print('第{}张图像处理完毕,耗时:{}秒'.format(num+1, round((time_end - t_start), 4)))

    t_end = time.time()
    print('所有图像处理完毕,耗时:{}秒'.format(round((t_end - t_start), 4)))