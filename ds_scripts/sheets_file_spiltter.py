#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.6.21
"""

import os

from myutils.project_utils import traverse_dir_files, read_file, write_list_to_file
from root_dir import DATA_DIR


class SheetsFileSplitter(object):
    """
    答题卡文件拆分
    """
    def __init__(self):
        self.sheets_dir = os.path.join(DATA_DIR, 'sheets')

    def process(self):
        """
        处理
        """
        print('[Info] 文件夹: {}'.format(self.sheets_dir))
        paths_list, names_list = traverse_dir_files(self.sheets_dir)
        print('[Info] 文本数: {}'.format(len(paths_list)))
        start_idx = 0
        gap = 200
        for path, name in zip(paths_list, names_list):
            print('[Info] path: {}'.format(path))
            name_x = name.split('.')[0]
            out_path = os.path.join(
                self.sheets_dir, '{}_{}_{}.txt'.format(name_x, str(start_idx), str(start_idx+gap)))
            data_lines = read_file(path)
            data_lines = data_lines[start_idx:start_idx+gap]
            print('[Info] 样本数量: {}'.format(len(data_lines)))
            write_list_to_file(out_path, data_lines)
            print('[Info] 写入完成: {}'.format(out_path))
        print('[Info] 拆分完成!')


def main():
    sfs = SheetsFileSplitter()
    sfs.process()


if __name__ == '__main__':
    main()
