#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 2.6.21
"""


import os
import sys

from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.cv4png_utils import *
from root_dir import DATA_DIR


class DbGenerator(object):

    def __init__(self):
        self.words_dict = DbGenerator.generate_words_dict()
        self.news_lines = DbGenerator.generate_news_lines()
        pass

    @staticmethod
    def clean_news_file():
        """
        清洗头条的新闻文本
        """
        file_path = os.path.join(DATA_DIR, 'toutiao_cat_data.txt')
        print('[Info] 输入文件: {}'.format(file_path))
        out_path = os.path.join(DATA_DIR, 'toutiao_cat_data.clean.txt')
        print('[Info] 输出文件: {}'.format(out_path))
        data_lines = read_file(file_path)

        news_list = []

        # 样本: 6552368441838272771_!_101_!_news_culture_!_发酵床的垫料种类有哪些？哪种更好？_!_
        for data_line in data_lines:
            items = data_line.split("_!_")
            news_str = items[3]
            news_list.append(news_str)

        print('[Info] 样本数: {}'.format(len(news_list)))
        write_list_to_file(out_path, news_list)

    @staticmethod
    def generate_news_lines(is_random=False):
        """
        创建新闻的文本行
        """
        news_file = os.path.join(DATA_DIR, 'toutiao_cat_data.clean.txt')
        news_lines = read_file(news_file)
        if not is_random:
            random.seed(47)
        random.shuffle(news_lines)
        print('[Info] 初始化新闻文本!')
        return news_lines

    @staticmethod
    def generate_words_dict():
        """
        创建文字字典
        """
        words_dir = os.path.join(DATA_DIR, 'chinese_word_files')
        paths_list, names_list = traverse_dir_files(words_dir)

        words_dict = dict()
        for path, name in zip(paths_list, names_list):
            word = name.split('.')[0]
            urls = read_file(path)
            words_dict[word] = urls

        print('[Info] 初始化字典完成!')
        return words_dict

    def get_word_png(self, word, idx=-1):
        """
        输入汉字，输出汉字对应的PNG图像
        """
        words_dict = self.words_dict
        if word not in words_dict:
            return

        urls = words_dict[word]
        n_word = len(urls)
        if 0 <= idx < n_word:
            url = urls[idx]
        else:
            idx = random.randint(0, n_word-1)
            url = urls[idx]
        _, img_bgr = download_url_img(url)

        img_new = resize_with_padding(img_bgr, 144)
        # show_img_bgr(img_new)

        img_bold = improve_img_bold(img_new, times=5)
        img_word = img_white_2_png(img_bold)
        return img_word

    def get_symbol_word(self):
        img = np.ones((144, 144, 3), np.uint8) * 255
        img = img.astype(np.uint8)

        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "!"
        font_scale = 4
        thickness = 4

        # get boundary of this text
        # text, fontFace, fontScale, thickness
        text_size = cv2.getTextSize(
            text=text, fontFace=font, fontScale=font_scale, thickness=thickness)[0]

        # get coords based on boundary
        text_x = int((img.shape[1] - text_size[0]) / 2)
        text_y = int((img.shape[0] + text_size[1]) / 2)

        # add text centered on image
        cv2.putText(img, text, (text_x, text_y), font,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        show_img_bgr(img)

    def get_news_line(self, idx=0):
        news_line = self.news_lines[idx]
        png_list = []
        self.get_symbol_word()
        # for word in news_line:
        #     def


def main():
    dg = DbGenerator()
    # dg.get_word_png("春", idx=5)
    # dg.get_word_img("美")
    dg.get_news_line()



if __name__ == '__main__':
    main()
