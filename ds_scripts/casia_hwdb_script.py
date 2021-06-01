#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 1.6.21
"""

import os
import sys
import numpy as np
import requests

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import Pool
from root_dir import DATA_DIR, ROOT_DIR
from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.cv4png_utils import *


class CasiaHwdbScript(object):
    def __init__(self):
        self.text_lines_file = os.path.join(DATA_DIR, 'casia_hwdb', 'line_label.txt')
        self.bkg_dir = os.path.join(DATA_DIR, 'casia_hwdb', 'bkgs', 'white_board')

    def load_bkgs(self):
        paths_list, names_list = traverse_dir_files(self.bkg_dir)
        bkg_list = []
        for path in paths_list:
            img_bgr = cv2.imread(path)
            bkg_list.append(img_bgr)
        return bkg_list

    def add_text_2_bkg(self):
        bkg_list = self.load_bkgs()
        bkg_img = bkg_list[0]
        bkg_img = cv2.resize(bkg_img, None, fx=2.0, fy=2.0)
        data_lines = read_file(self.text_lines_file)
        for data_line in data_lines:
            url, text = data_line.split("\t")
            print('[Info] url: {}'.format(url))
            _, img_gray = download_url_img(url)
            img_gray = cv2.resize(img_gray, None, fx=0.3, fy=0.3)
            show_img_bgr(img_gray)
            print('[Info] img_bgr: {}'.format(img_gray.shape))
            img_mask = np.where(img_gray == 255, 0, 255)
            img_bgra = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGRA)
            img_bgra[:, :, 3] = img_mask
            show_img_png(img_bgra, save_name="xxx.png")
            xxx_img = paste_png_on_bkg(img_bgra, bkg_img, (800, 500))
            show_img_bgr(xxx_img, save_name="xxx.jpg")
            break

    def traverse_word_folder(self):
        from x_utils.oss_utils import traverse_oss_folder, traverse_and_save_oss_folder
        out_file = "../mydata/chinese_word_628841.txt"
        item_url = traverse_and_save_oss_folder("jiade/edu/手写/公开数据集/CASIA/最常用的1000个汉字/", out_file, 'png')
        print('[Info] url: {}'.format(len(item_url)))

    def process_words_file(self):
        words_path = os.path.join(DATA_DIR, 'chinese_word_628841.txt')
        out_dir = os.path.join(DATA_DIR, 'chinese_word_files')
        mkdir_if_not_exist(out_dir)
        word_urls = read_file(words_path)

        word_urls_dict = collections.defaultdict(list)
        for word_url in word_urls:
            word_name = word_url.split('/')[-2]
            word_urls_dict[word_name].append(word_url)

        for word_name in word_urls_dict.keys():
            out_file = os.path.join(out_dir, word_name+".txt")
            urls = word_urls_dict[word_name]
            write_list_to_file(out_file, urls)

        print('[Info] 处理完成: {}'.format(out_dir))


    @staticmethod
    def process_lines(data_lines, out_dir, name_x):
        print('[Info] {} 总数: {}'.format(name_x, len(data_lines)))
        for idx, data_line in enumerate(data_lines):
            out_file = os.path.join(out_dir, "{}_{}.png".format(name_x, idx))
            img_data = requests.get(data_line).content
            print(data_line)
            with open(out_file, 'wb') as hl:
                hl.write(img_data)

            if idx % 100 == 0:
                print('[Info] {} count {}'.format(name_x, idx))

        print('[Info] 处理完成: {}'.format(name_x))

    def download_imgs(self):
        files_dir = os.path.join(DATA_DIR, 'chinese_word_files')

        out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'chinese_words_1000')
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(files_dir)

        pool = Pool(processes=20)
        for path, name in zip(paths_list, names_list):
            name_x = name.split('.')[0]
            word_dir = os.path.join(out_dir, name_x)
            mkdir_if_not_exist(word_dir)
            data_lines = read_file(path)
            # CasiaHwdbScript.process_lines(data_lines, word_dir, name_x)
            pool.apply_async(CasiaHwdbScript.process_lines, (data_lines, word_dir, name_x))

        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(out_dir))




def main():
    chs = CasiaHwdbScript()
    # chs.add_text_2_bkg()
    # chs.traverse_word_folder()
    # chs.process_words_file()
    chs.download_imgs()


if __name__ == '__main__':
    main()
