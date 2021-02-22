# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/22

import cv2
import numpy as np


points = np.array([[342,230],[342,192],[378,192],[378, 230]])

res = cv2.minAreaRect(points)

print(res)
