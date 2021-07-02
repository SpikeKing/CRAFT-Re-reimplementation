#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 1.7.21
"""

import os
import json
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import Pool
from x_utils.vpf_utils import get_english_words_cutter_service
from myutils.project_utils import *
from root_dir import DATA_DIR


class EnWordsProcessor(object):
    def __init__(self):
        self.imgs_file_path = os.path.join(DATA_DIR, '第二批_高新+英语649.txt')
        self.tmp_file_path = os.path.join(DATA_DIR, '第二批_高新+英语649.tmp.txt')
        self.out_file_path = os.path.join(DATA_DIR, '第二批_高新+英语649.out.txt')
        self.error_file_path = os.path.join(DATA_DIR, '第二批_高新+英语649.error.txt')

    @staticmethod
    def process_url(idx, url, out_path, err_path):
        try:
            res_dict = get_english_words_cutter_service(url)
            data_dict = res_dict["data"]
            res_str = json.dumps(data_dict)
            write_line(out_path, res_str)
            print('[Info] idx: {}'.format(idx))
        except Exception as e:
            write_line(err_path, url)
            print('[Info] idx: err: {}'.format(url))

    @staticmethod
    def process_url_call(data):
        return EnWordsProcessor.process_url(data[0], data[1], data[2], data[3])

    def process(self):
        data_lines = read_file(self.imgs_file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        out_list, err_list = [], []
        for idx, data_line in enumerate(data_lines):
            print('[Info] data_line: {}'.format(data_line))
            url = data_line
            # try:
            res_dict = get_english_words_cutter_service(url)
            print(res_dict)
            data_dict = res_dict["data"]
            res_str = json.dumps(data_dict)

            out_list.append(res_str)
            write_line(self.tmp_file_path, res_str)
            print('[Info] idx: {}'.format(idx))
            # except Exception as e:
            #     err_list.append(url)
            #     print('[Info] err: {}'.format(url))

        print('[Info] 数据: {}'.format(len(out_list)))
        write_list_to_file(self.out_file_path, out_list)
        print('[Info] 写入完成: {}'.format(self.out_file_path))

    def process_mp(self):
        # create_file(self.out_file_path)
        data_lines = read_file(self.imgs_file_path)
        data_list = []
        for idx, data_line in enumerate(data_lines):
            data_list.append([idx, data_line, self.out_file_path, self.error_file_path])

        pool = Pool(processes=5)
        pool.map(EnWordsProcessor.process_url_call, data_list)
        pool.close()
        pool.join()
        print('[Info] 全部处理完成: {}'.format(self.out_file_path))

    def merge_file(self):
        file1_path = os.path.join(DATA_DIR, '第一批_高新-英语-645.out.txt')
        file2_path = os.path.join(DATA_DIR, '第二批_高新+英语649.out.txt')
        out_path = os.path.join(DATA_DIR, '四线格1000.txt')
        data_lines1 = read_file(file1_path)
        data_lines2 = read_file(file2_path)
        data_lines = data_lines1 + data_lines2
        print('[Info] data_lines: {}'.format(len(data_lines)))
        data_lines = data_lines[:1000]
        print('[Info] data_lines: {}'.format(len(data_lines)))
        write_list_to_file(out_path, data_lines)


def main():
    ewp = EnWordsProcessor()
    # ewp.process_mp()
    ewp.process()
    # ewp.merge_file()


if __name__ == '__main__':
    main()
