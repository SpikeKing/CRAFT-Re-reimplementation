#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 2.6.21
"""

import base64
import numpy as np
import cv2
from myutils.cv_utils import show_img_bgr


def main1():
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


def main2():
    image_encode = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAVABUDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDQ+KHxYubX48eHfhJoPjeDSfLtpNS1mWcIslwrxyRW9nG75EfmOskjOpRwLRVBPmYHJeIPGvxf0b4naRoWq6Z4fvU1q9knjj00z28kUCSjdJJNIx3MPMDFD8x2sQcYzrftc6F4GvNR8NeJ/HPxNn0PT7DVUll0bSbGAX3iO5GVtoA0hJkSJ3YmMKwzOCSh2Ey+KrC98T/HPR4bu1UPbeHry5lgVd8ls7vBGfMUHlsu4zzt9TiuxTklY5Oax00GsPIm6UqfRgCM++DyPoeaKqadI99CZTp7K4bbJvk/iHXBAOf5+tFCqSaFzM8r8d/E/wCFPjPxmnjT4h/Am08Q3H9kR2Nrb6xqCTwWce95HaFHgIikdmTe64ZhBEM/IK5y5174B6XdSS+GfgPd+HpWIYTeFfGl1pZXuBi2VVZRx8rAqSoJBOclFaQjFx2LSVjO1L9rHxH8ONWfTdX0m419Lq2jntbq/wBT2XMaEuojkdIwspG0fOEQnvk80UUU+SHYLI//2Q=="
    image_content = base64.urlsafe_b64decode(image_encode)
    nparr = np.asarray(bytearray(image_content)).reshape(1, -1)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)
    show_img_bgr(img)

if __name__ == '__main__':
    main2()
