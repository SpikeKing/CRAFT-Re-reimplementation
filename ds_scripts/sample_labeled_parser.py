#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.6.21
"""

import cv2
import os
import json
import collections

from xml.dom.minidom import parse
import xml.dom.minidom

from myutils.project_utils import write_list_to_file, unfold_nested_list
from myutils.cv_utils import check_point_in_box
from root_dir import DATA_DIR


class SampleLabeledParser(object):
    """
    简单样本解析
    """
    def __init__(self):
        self.image_dir = os.path.join(DATA_DIR, 'tmps')
        self.label_path = os.path.join(DATA_DIR, 'tmps', 'annotations.xml')
        self.out_labeled = os.path.join(DATA_DIR, 'tmps', 'out_labeled.txt')

    @staticmethod
    def split_boxes(pnt_list, box):
        """
        根据点列表拆分box
        """
        if not pnt_list:
            return [box]
        x_min, y_min, x_max, y_max = box
        x_list = []
        for pnt in pnt_list:
            x_list.append(pnt[0])
        x_list = sorted(x_list)
        sub_boxes = []
        x_s = x_min
        for x in x_list:
            sub_boxes.append([x_s, y_min, x, y_max])
            x_s = x
        sub_boxes.append([x_s, y_min, x_max, y_max])
        return sub_boxes

    @staticmethod
    def parse_pnt_and_box(box_pnt_dict, box_list, img_bgr=None):
        """
        解析点和box
        """
        sub_boxes_list = []

        for idx in box_pnt_dict.keys():
            pnt_list = box_pnt_dict[idx]
            # print('[Info] pnt_list: {}'.format(pnt_list))
            box = box_list[idx]
            sub_boxes = SampleLabeledParser.split_boxes(pnt_list, box)
            sub_boxes_list.append(sub_boxes)

        sub_boxes_list = unfold_nested_list(sub_boxes_list)  # 双层list变成单层list

        # 划掉文字的区域需要区分对待
        for x_idx in range(len(box_list)):
            if x_idx not in box_pnt_dict.keys():
                sub_boxes_list.append(box_list[x_idx])

        # tmp_path = os.path.join(DATA_DIR, 'tmps', 'sub_boxes.jpg')
        # draw_box_list(img_bgr, sub_boxes_list, is_text=False, color=(255, 0, 0), save_name=tmp_path)
        return sub_boxes_list

    def process_annotations(self):
        """
        处理解析标签
        """
        DOMTree = xml.dom.minidom.parse(self.label_path)
        collection = DOMTree.documentElement
        meta = collection.getElementsByTagName("meta")
        # print('[Info] meta: {}'.format(meta))
        image_data = collection.getElementsByTagName("image")
        print('[Info] 样本数: {}'.format(len(image_data)))

        anno_list = []  # 标签信息列表
        for image in image_data:
            image_name = image.getAttribute("name")
            print('[Info] image: {}'.format(image_name))
            img_bgr = cv2.imread(os.path.join(self.image_dir, image_name))
            print('[Info] img_bgr: {}'.format(img_bgr.shape))
            box_data = image.getElementsByTagName("box")
            points_data = image.getElementsByTagName("points")
            print('[Info] box_data: {}'.format(len(box_data)))
            print('[Info] points_data: {}'.format(len(points_data)))
            box_list = []
            for box in box_data:
                x_min = float(box.getAttribute("xtl"))
                y_min = float(box.getAttribute("ytl"))
                x_max = float(box.getAttribute("xbr"))
                y_max = float(box.getAttribute("ybr"))
                box = [x_min, y_min, x_max, y_max]
                box = [int(x) for x in box]
                box_list.append(box)
            print('[Info] 框数量: {}'.format(len(box_list)))
            # tmp_path = os.path.join(DATA_DIR, 'tmps', 'boxes.jpg')
            # draw_box_list(img_bgr, box_list, is_text=False, color=(255, 0, 0), save_name=tmp_path)
            box_pnt_dict = collections.defaultdict(list)
            for points in points_data:
                pnt_str = points.getAttribute("points")
                pnt_list = pnt_str.split(",")
                pnt = [int(float(x)) for x in pnt_list]
                is_inside = False
                for idx, box in enumerate(box_list):
                    if check_point_in_box(pnt, box):
                        box_pnt_dict[idx].append(pnt)
                        is_inside = True
                        break
                if not is_inside:
                    print('[Info] error pnt: {}'.format(pnt))

            sub_boxes_list = self.parse_pnt_and_box(box_pnt_dict, box_list, img_bgr=img_bgr)
            print('[Info] 全部框数: {}'.format(len(sub_boxes_list)))
            img_anno_dict = {
                "image_name": image_name,
                "char_boxes": sub_boxes_list
            }
            img_anno_str = json.dumps(img_anno_dict)
            anno_list.append(img_anno_str)
        print('[Info] 标签数量: {}'.format(len(anno_list)))
        write_list_to_file(self.out_labeled, anno_list)
        print('[Info] 标签文本写入完成: {}'.format(self.out_labeled))


def main():
    slp = SampleLabeledParser()
    slp.process_annotations()


if __name__ == '__main__':
    main()
