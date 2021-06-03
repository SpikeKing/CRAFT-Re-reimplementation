#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 2.6.21
"""

import numpy as np
import cv2
from myutils.cv_utils import show_img_bgr


def main():
    img = np.zeros((600, 1000, 3), np.uint8)

    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Hello Joseph!!"

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)

    # add text centered on image
    cv2.putText(img, text, (textX, textY ), font, 1, (255, 255, 255), 2)
    show_img_bgr(img)


if __name__ == '__main__':
    main()
