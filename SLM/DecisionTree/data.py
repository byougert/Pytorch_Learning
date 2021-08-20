# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/12 18:16

import pandas


def load_data(data_path, encoding, header):
    df = pandas.read_csv(data_path, encoding=encoding, header=header)
    title = list(df)
    return df, title
