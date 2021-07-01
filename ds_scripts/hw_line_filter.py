#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 25.6.21
"""

import os

from x_utils.vpf_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class HwLineFilter(object):
    def __init__(self):
        self.image_path = os.path.join(DATA_DIR, 'sheets', 'sheet_7_biology_0_200.txt')
        self.hw_lines_path = os.path.join(DATA_DIR, 'train.txt')
        self.out_hw_lines_path = os.path.join(DATA_DIR, 'hw_lines.out.txt')

    @staticmethod
    def is_chinese(uchar):
        """
        判断中文
        """
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    @staticmethod
    def ratio_chinese_of_line(line):
        u_line = unicode_str(line)
        num_of_c = 0
        num_of_line = len(line)
        for char in u_line:
            if HwLineFilter.is_chinese(char):
                num_of_c += 1
        # print('[Info] u_line: {}'.format(u_line))
        # print('[Info] 中文数: {}, 行字: {}'.format(num_of_c, num_of_line))
        ratio = safe_div(num_of_c, num_of_line)
        # print('[Info] ratio: {}'.format(ratio))
        return ratio

    def process(self):
        data_lines = read_file(self.hw_lines_path)

        chinese_lines = []
        for idx, data_line in enumerate(data_lines):
            if idx % 1000 == 0:
                print('[Info] idx: {}'.format(idx))
            items = data_line.split("\t")
            # print('[Info] items: {}'.format(len(items)))
            url = items[0]
            label = items[1]
            # print('[Info] label: {}'.format(label))
            c_ratio = self.ratio_chinese_of_line(label)
            if c_ratio > 0.5:
                chinese_lines.append(data_line)
            if len(chinese_lines) % 1000 == 0:
                print('[Info] chinese_lines: {}'.format(len(chinese_lines)))
        print('[Info] 中文行数: {}'.format(len(chinese_lines)))
        create_file(self.out_hw_lines_path)
        write_list_to_file(self.out_hw_lines_path, chinese_lines)
        print('[Info] 写入完成: {}'.format(self.out_hw_lines_path))


def main():
    hlf = HwLineFilter()
    hlf.process()


if __name__ == '__main__':
    main()
