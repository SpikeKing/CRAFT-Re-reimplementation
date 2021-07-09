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
    def process_line(idx, data, img_url, out_file, err_file):
        print('[Info] img_url: {}, idx: {}'.format(img_url, idx))
        try:
            _, img_bgr = download_url_img(img_url)
            img_name = img_url.split('/')[-1]
            pos_data = data['pos']
            # print('[Info] pos: {}'.format(pos_data))
            img_name = "{}_{}.jpg".format(img_name.split('.')[0], idx)
            print('[Info] img_name: {}'.format(img_name))
            rec_box = EnWordsProcessorV2.parse_pos_2_rec(pos_data)
            bbox = rec2bbox(rec_box)
            img_patch = get_cropped_patch(img_bgr, bbox)
            img_patch_url = EnWordsProcessorV2.save_img_patch(img_patch, img_name)
            print('[Info] img_patch_url: {}'.format(img_patch_url))
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
    def process_line_core(data):
        return EnWordsProcessorV2.process_line(data[0], data[1], data[2], data[3], data[4])

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
            # print('[Info] data: {}'.format(data_list))
            for idx, data in enumerate(data_list):
                param = [idx, data, img_url, self.out_file, self.err_file]
                param_list.append(param)
            # if img_idx == 10:
            #     break
            print('[Info] img_idx: {}'.format(img_idx))
        print('[Info] 样本处理完成: {}'.format(len(param_list)))

        # for param in param_list:
        #     EnWordsProcessorV2.process_line_process(param)

        pool = Pool(processes=5)
        pool.map(EnWordsProcessorV2.process_line_core, param_list)
        pool.close()
        pool.join()
        print('[Info] 全部处理完成: {}'.format(self.out_file))

    def check_data(self):
        """
        检查数据
        """
        self.data_path = os.path.join(DATA_DIR, '20210702_行挑选.out.20210705094001.txt')
        print('[Info] 检查数据: {}'.format(self.data_path))
        data_lines = read_file(self.data_path)
        print('[Info] 数量: {}'.format(len(data_lines)))
        for data_line in data_lines[1:]:
            print('[Info] data_line: {}'.format(data_line))
            data_dict = json.loads(data_line)
            image_url = data_dict['image_url']
            image_original_url = data_dict['image_original_url']
            print('[Info] image_url: {}'.format(image_url))
            print('[Info] image_original_url: {}'.format(image_original_url))
            boxes = data_dict['boxes']
            _, img_bgr = download_url_img(image_url)
            draw_rec_list(img_bgr, boxes, is_text=False, save_name="tmp.jpg")
            break

    def process_v2(self):
        """
        整体处理逻辑
        """
        print('[Info] 输入文件路径: {}'.format(self.file_path))
        data_line = read_file(self.file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] data_dict: {}'.format(len(data_dict)))
        param_list = []

        for img_idx, img_url in enumerate(data_dict.keys()):
            data_list = data_dict[img_url]
            bboxes = []
            for idx, data in enumerate(data_list):
                param = [idx, data, img_url, self.out_file, self.err_file]
                param_list.append(param)
                pos_data = data['pos']
                rec_box = EnWordsProcessorV2.parse_pos_2_rec(pos_data)
                bbox = rec2bbox(rec_box)
                bboxes.append(bbox)
            if len(bboxes) != 1:
                continue
        print('[Info] img_url: {}'.format(img_url))
        print('[Info] bboxes: {}'.format(bboxes))
        _, img_bgr = download_url_img(img_url)
        draw_box_list(img_bgr, bboxes, thickness=-1, is_show=True, is_overlap=True, save_name="xxx.jpg")


def main():
    ewp = EnWordsProcessorV2()
    # ewp.process()
    # ewp.check_data()
    ewp.process_v2()


if __name__ == '__main__':
    main()
