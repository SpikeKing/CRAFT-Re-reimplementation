#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 3.6.21
"""
import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import Pool

from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.oss_utils import *
from x_utils.vpf_utils import *


class LabelPrepare(object):
    """
    准备标注数据
    """
    def __init__(self):
        self.label_folder = os.path.join(DATA_DIR, 'label_files')

    @staticmethod
    def process_url(img_url, url_name_dict):
        """
        处理URL
        """
        oss_folder = "zhengsheng.wcl/Character-Detection/datasets/k12-images-yuwen-grade1_4/"
        img_name = img_url.split("/")[-1].lower()
        res_dict = get_trt_rotation_vpf_service(img_url)
        rotated_image_url = res_dict['data']['rotated_image_url']
        _, img_bgr = download_url_img(rotated_image_url)
        folder_name = url_name_dict[img_url]
        save_img_2_oss(img_bgr, folder_name + img_name, oss_folder)

    def rotate_img_urls(self):
        """
        旋转URL
        """
        print('[Info] 处理文件: {}'.format(self.label_folder))
        paths_list, names_list = traverse_dir_files(self.label_folder)
        urls_list = []  # 添加urls
        url_name_dict = dict()  # url和name的字典
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)
            urls_list += data_lines
            for data_line in data_lines:
                url_name_dict[data_line] = name.split(".")[0]

        print('[Info] 样本数: {}'.format(len(urls_list)))

        pool = Pool(processes=100)
        for idx, img_url in enumerate(urls_list):
            print('[Info] idx: {}, url: {}'.format(idx, img_url))
            try:
                # LabelPrepare.process_url(img_url, url_name_dict)
                pool.apply_async(LabelPrepare.process_url, (img_url, url_name_dict))
            except Exception as e:
                print('[Exception] e: {}'.format(e))
                print('[Exception] url: {}'.format(img_url))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.label_folder))

    @staticmethod
    def generate_urls_files():
        file_dict = {
            "1年级/上学期/语文/杨国旗-第四批/": "grade_1up_yuwen_yangguoqi",
            "1年级/上学期/语文/陈玉凡/": "grade_1up_yuwen_chenyufan",
            "2年级/上学期/语文/杨国旗-第四批/": "grade_2up_yuwen_yangguoqi",
            "2年级/上学期/语文/陈玉凡/": "grade_2up_yuwen_chenyufan",
            "2年级/下学期/语文/杨国旗-第四批/": "grade_2down_yuwen_yangguoqi",
            "2年级/下学期/语文/陈玉凡/": "grade_2down_yuwen_chenyufan",
            "3年级/上学期/语文/杨国旗-第四批/": "grade_3up_yuwen_yangguoqi",
            "3年级/上学期/语文/陈玉凡/": "grade_3up_yuwen_chenyufan",
            "4年级/上学期/语文/贺飞帆/四年级上册语文卷子/": "grade_4up_yuwen_hefeifan",
            "4年级/上学期/语文/陈玉凡/": "grade_4up_yuwen_chenyufan",
            "4年级/上学期/语文/高新/": "grade_4up_yuwen_gaoxin",
            "4年级/下学期/语文/杨国旗-第四批/": "grade_4down_yuwen_yangguoqi",
        }

        for oss_name in file_dict.keys():
            oss_dir = "jiade/edu/手写/原始数据/k12-image/{}".format(oss_name)
            file_name = file_dict[oss_name] + ".txt"
            out_file = os.path.join(DATA_DIR, 'label_files', file_name)
            create_file(out_file)
            download_oss_dir(oss_dir, out_file, 'jpg')

        print('[Info] 处理完成!')

    @staticmethod
    def generate_clean_urls_files():
        """
        读取oss的file
        """
        oss_name = "k12-images-yuwen-grade1_4"
        oss_dir = "zhengsheng.wcl/Character-Detection/datasets/{}/".format(oss_name)
        file_name = oss_name + ".txt"
        out_file = os.path.join(DATA_DIR, file_name)
        create_file(out_file)
        download_oss_dir(oss_dir, out_file, 'jpg')

        print('[Info] 处理完成!')

    @staticmethod
    def split_url_files():
        file_name = "k12-images-yuwen-grade1_4"
        urls_file = os.path.join(DATA_DIR, '{}.txt'.format(file_name))
        out_dir = os.path.join(DATA_DIR, 'label_clean_files')
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(urls_file)
        gap = 2000
        for idx in range(0, len(data_lines), gap):
            out_file = os.path.join(out_dir, '{}_{}_{}.txt'.format(file_name, idx, idx+gap))
            create_file(out_file)
            urls_list = data_lines[idx:idx+gap]
            write_list_to_file(out_file, urls_list)
        print('[Info] 处理完成: {}'.format(urls_file))




def main():
    lp = LabelPrepare()
    # lp.rotate_img_urls()
    # lp.generate_urls_files()
    lp.split_url_files()


if __name__ == '__main__':
    main()
