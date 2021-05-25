#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 25.5.21
"""
import itertools
import os
import re

import cv2
import scipy.io as sci_io

from myutils.cv_utils import show_img_bgr, draw_rec_list
from root_dir import DATA_DIR


class SynthTextChecker(object):
    """
    检测数据集的格式
    """
    def __init__(self):
        pass

    def check_sample(self):
        print('[Info] check sample')
        synth_text_folder = os.path.join(DATA_DIR, 'SynthText')
        data_path = os.path.join(synth_text_folder, 'gt.mat')
        gt_data = sci_io.loadmat(data_path)
        print('[Info] data: {}'.format(type(gt_data)))
        print('[Info] data: {}'.format(gt_data.keys()))
        char_boxes = gt_data['charBB'][0]
        word_boxes = gt_data['wordBB'][0]
        img_names = gt_data['imnames'][0]
        img_texts = gt_data['txt'][0]
        print('[Info] char_box {}'.format(char_boxes.shape))
        print('[Info] image {}'.format(len(img_names)))
        print('[Info] img_txt {}'.format(len(img_texts)))

        sample_char_box = char_boxes[0]
        sample_char_box = sample_char_box.transpose((2, 1, 0))  # 转换位置
        sample_word_box = word_boxes[0]
        sample_word_box = sample_word_box.transpose((2, 1, 0))
        sample_img_name = img_names[0]
        sample_img_text = img_texts[0]
        print('[Info] sample_char_box: {}'.format(sample_char_box.shape))
        print('[Info] sample_word_box: {}'.format(sample_word_box.shape))
        print('[Info] sample_img_name: {}'.format(sample_img_name))
        print('[Info] sample_img_text: {}'.format(sample_img_text))

        rec_list = []
        for word_box in sample_word_box:
            rec_list.append(word_box.astype(int).tolist())

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in sample_img_text]
        words = list(itertools.chain(*words))  # 二维数组转换为1维
        print('[Info] num of words: {}'.format(len(words)))
        print('[Info] words: {}'.format(words))

        img_path = os.path.join(synth_text_folder, sample_img_name[0])
        img_bgr = cv2.imread(img_path)
        show_img_bgr(img_bgr)

        draw_rec_list(img_bgr, rec_list, is_show=True)

        print('[Info] 处理完成: {}'.format(data_path))

        pass


def main():
    stc = SynthTextChecker()
    stc.check_sample()


if __name__ == '__main__':
    main()
