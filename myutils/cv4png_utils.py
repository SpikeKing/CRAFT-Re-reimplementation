#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 1.6.21
"""

import cv2
import numpy as np


def show_img_png(img_bgra, save_name=None):
    """
    展示BGRA的PNG图
    """
    import matplotlib.pyplot as plt

    img_rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    plt.imshow(img_rgba)
    plt.show()

    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_rgba)


def paste_png_on_bkg(draw_png, bkg_png, offset):
    """
    PNG粘贴到背景之上，支持处理3通道的背景
    """
    bh, bw, bc = bkg_png.shape
    if bc == 3:
        bkg_png = cv2.cvtColor(bkg_png, cv2.COLOR_BGR2BGRA)

    h, w, _ = draw_png.shape
    x, y = offset

    alpha_mask = np.where(draw_png[:, :, 3] == 255, 1, 0)
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)  # 将mask复制4次

    y_s, y_e = min(y, bh), min(y + h, bh)
    x_s, x_e = min(x, bw), min(x + w, bw)

    bkg_png[y_s:y_e, x_s:x_e, :] = (1.0 - alpha_mask) * bkg_png[y_s:y_e, x_s:x_e] + alpha_mask * draw_png

    if bc == 3:
        bkg_png = cv2.cvtColor(bkg_png, cv2.COLOR_BGRA2BGR)

    return bkg_png
