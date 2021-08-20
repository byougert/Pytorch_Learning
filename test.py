# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/6 15:26

from PIL import Image
import numpy as np


def main():
    path = r'F:\PythonSpace\Pytorch_Learning\pytorch\tutorial\image_and_video\Object_Detection\data\PennFudanPed\PedMasks\FudanPed00001_mask.png'
    mask = np.array(Image.open(path))
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if mask[i][j] == 1:
                mask[i][j] = 254
            elif mask[i][j] == 2:
                mask[i][j] = 120
    mask = Image.fromarray(np.uint8(mask))
    mask.show()


if __name__ == '__main__':
    main()