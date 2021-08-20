# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/3/8 20:11

import logging

__fmt = r'%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
__date_format = r'%m/%d/%Y %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=__fmt, datefmt=__date_format)
logger = logging.getLogger(__name__)
