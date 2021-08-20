# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/29 18:42

from pytorch.tutorial.image_and_video.Object_Detection.data import PennFudanDataset


dataset = PennFudanDataset(r'data\PennFudanPed', None)
print(dataset[0])