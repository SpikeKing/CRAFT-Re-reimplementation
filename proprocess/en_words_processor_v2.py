#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 1.7.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.vpf_utils import get_english_words_cutter_service
from x_utils.oss_utils import save_img_2_oss
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class EnWordsProcessorV2(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, '20210702_行挑选.json')
        time_str = get_current_time_str()
        self.out_file = os.path.join(DATA_DIR, '20210702_行挑选.out.{}.txt'.format(time_str))
        self.err_file = os.path.join(DATA_DIR, '20210702_行挑选.error.{}.txt'.format(time_str))

    @staticmethod
    def parse_pos_2_rec(pos_data):
        """
        解析pos
        """
        pos_list = []
        for pos in pos_data:
            x = pos['x']
            y = pos['y']
            pos_list.append([x, y])
        return pos_list

    @staticmethod
    def save_img_patch(img_bgr, img_name):
        """
        上传图像
        """
        oss_root_dir = "zhengsheng.wcl/Character-Detection/datasets/english-words-patch-20210702/"
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_line(idx, img_bgr, data, img_url, out_file, err_file):
        print('[Info] idx: {}'.format(idx))
        try:
            img_name = img_url.split('/')[-1]
            pos_data = data['pos']
            # print('[Info] pos: {}'.format(pos_data))
            img_name = "{}_{}.jpg".format(img_name.split('.')[0], idx)
            rec_box = EnWordsProcessorV2.parse_pos_2_rec(pos_data)
            bbox = rec2bbox(rec_box)
            img_patch = get_cropped_patch(img_bgr, bbox)
            img_patch_url = EnWordsProcessorV2.save_img_patch(img_patch, img_name)
            # print('[Info] img_patch_url: {}'.format(img_patch_url))
            res_dict = get_english_words_cutter_service(img_patch_url)
            # print('[Info] res_dict: {}'.format(res_dict))
            data_dict = res_dict["data"]
            # print('[Info] data_dict: {}'.format(data_dict))
            data_dict['image_original_url'] = img_url
            data_str = json.dumps(data_dict)
            write_line(out_file, data_str)
            print('[Info] 处理完成: {}'.format(idx))
        except Exception as e:
            print('[Error] 失败: {}'.format(idx))
            write_line(err_file, img_url)

    @staticmethod
    def process_line_process(data):
        return EnWordsProcessorV2.process_line(data[0], data[1], data[2], data[3], data[4], data[5])

    def process(self):
        """
        处理
        """
        print('[Info] 输入文件路径: {}'.format(self.file_path))
        data_line = read_file(self.file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] data_dict: {}'.format(len(data_dict)))

        param_list = []
        for img_idx, img_url in enumerate(data_dict.keys()):
            # print('[Info] img_url: {}'.format(img_url))
            data_list = data_dict[img_url]
            _, img_bgr = download_url_img(img_url)
            # print('[Info] data: {}'.format(data_list))
            for idx, data in enumerate(data_list):
                tag_str = "{}_{}".format(img_url, idx)
                param = [tag_str, img_bgr, data, img_url, self.out_file, self.err_file]
                param_list.append(param)
            if img_idx == 10:
                break
            print('[Info] img_idx: {}'.format(img_idx))
        print('[Info] 样本处理完成: {}'.format(len(param_list)))

        for param in param_list:
            EnWordsProcessorV2.process_line_process(param)

        # pool = Pool(processes=5)
        # pool.map(EnWordsProcessorV2.process_line_process, param_list)
        # pool.close()
        # pool.join()
        print('[Info] 全部处理完成: {}'.format(self.out_file))


def main():
    ewp = EnWordsProcessorV2()
    ewp.process()


if __name__ == '__main__':
    main()
