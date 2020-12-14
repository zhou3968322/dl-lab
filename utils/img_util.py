# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import numpy as np


def is_box_outside_rect(poly, rect):
    """
    poly 是否在rect外, true 为outside

    """
    poly = np.array(poly)
    xl, yt, xr, yd = rect
    if len(poly.shape) == 2:
        if poly[:, 0].max() <= xl or poly[:, 0].min() >= xr:
            return True
        if poly[:, 1].max() <= yt or poly[:, 1].min() >= yd:
            return True
        return False
    else:
        x0, y0, x1, y1 = poly
        if x1 <= xl or x0 >= xr:
            return True
        if y1 <= yt or y0 >= yd:
            return True
        return False


def is_box_inside_rect(poly, rect):
    """
    poly 是否完全在rect内, true 为outside

    """
    poly = np.array(poly)
    xl, yt, xr, yd = rect  # [xl, xr)左闭右开
    if len(poly.shape) == 2:
        if poly[:, 0].min() <= xl or poly[:, 0].max() >= xr:
            return False
        if poly[:, 1].min() <= yt or poly[:, 1].max() >= yd:
            return False
        return True
    else:
        x0, y0, x1, y1 = poly
        if x0 <= xl or x1 >= xr:
            return False
        if y0 <= yt or y1 >= yd:
            return False
        return True


def convert_rect_to_polygon(rect):
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x0, y1],
                     [x1, y1], [x1, y0]])


def convert_poly_to_rect(poly):
    poly = np.array(poly)
    x0 = np.min(poly[:, 0])
    x1 = np.max(poly[:, 0])
    y0 = np.min(poly[:, 1])
    y1 = np.max(poly[:, 1])
    return np.array([x0, y0, x1, y1])