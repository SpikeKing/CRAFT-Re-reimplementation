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


def improve_img_bold(img_bgr, times=2.5):
    """
    加粗图像中的黑字部分
    """
    if times != 0:
        scale = 1.0 / float(times)
    else:
        return img_bgr
    img_mask = np.where(img_bgr[:] == (255, 255, 255), 0, 1)
    img_bgr[np.where(img_mask[:] == 1)] = \
        (img_bgr[np.where(img_mask[:] == 1)] * scale).astype(np.uint8)
    img_bgr = np.clip(img_bgr, 0, 255)
    return img_bgr


def img_white_2_png(img_bgr, bkg_color=(255, 255, 255), is_white=True):
    """
    白色图像转换为PNG图像
    """
    if is_white:
        img_char = img_bgr
    else:
        img_char = (img_bgr * (-1) + 255).astype(np.uint8)
    h, w, _ = img_bgr.shape
    img_mask = np.where(img_bgr[:] == bkg_color, 0, 255)
    img_alpha = img_mask[:, :, 2]  # 只使用最后一维
    img_new = np.zeros((h, w, 4), dtype=np.uint8)
    img_new[:, :, :3] = img_char
    img_new[:, :, 3] = img_alpha
    return img_new
