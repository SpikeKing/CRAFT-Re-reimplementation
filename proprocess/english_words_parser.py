#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 30.6.21
"""

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class EnglishWordsParser(object):
    """
    英语单词解析
    """
    def __init__(self):
        self.hw_templates = os.path.join(DATA_DIR, 'en_sheet_templates')
        self.hw_zone_path = os.path.join(self.hw_templates, 'instances_default.json')
        self.template_boxes_dict = self.parse_hw_zone(self.hw_zone_path, self.hw_templates)  # 加载模板bbox
        self.sheet_7_english = os.path.join(DATA_DIR, 'sheet_7_english')

    @staticmethod
    def parse_hw_zone(hw_zone_path, hw_templates):
        """
        生成区域模板
        """
        paths_list, names_list = traverse_dir_files(hw_templates)
        print('[Info] 模板数量: {}'.format(len(paths_list)))
        name_dict = {}
        for path in paths_list:
            name = path.split('/')[-1]
            if name.endswith('jpg'):
                name_dict[name] = path
        # print('[Info] name_dict: {}'.format(name_dict))

        data_lines = read_file(hw_zone_path)
        data_str = data_lines[0]
        data_dict = json.loads(data_str)
        # print('[Info] data_dict: {}'.format(data_dict.keys()))
        images = data_dict['images']
        # print('[Info] images: {}'.format(images))
        annotations = data_dict['annotations']
        # print('[Info] annotations: {}'.format(annotations))
        images_dict = dict()
        for image_data in images:
            image_id = image_data['id']
            file_name = image_data['file_name']
            images_dict[image_id] = file_name

        # print('[Info] annotations: {}'.format(len(annotations)))

        boxes_dict = collections.defaultdict(list)
        for anno in annotations:
            # print('[Info] anno: {}'.format(anno))
            image_id = anno['image_id']
            bbox = [int(x) for x in anno['bbox']]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes_dict[image_id].append(bbox)

        template_boxes_dict = dict()
        for image_idx in images_dict.keys():
            image_name = images_dict[image_idx]
            image_path = name_dict[image_name]
            bboxes = boxes_dict[image_idx]
            print('[Info] image_path: {}'.format(image_path))
            print('[Info] bboxes: {}'.format(len(bboxes)))
            # img_bgr = cv2.imread(image_path)
            # draw_box_list(img_bgr, bboxes, is_show=True)
            image_x = image_name.split('.')[0]
            template_boxes_dict[image_x] = bboxes

        return template_boxes_dict

    @staticmethod
    def download_sheets():
        name = "sheet_7_english"
        sheet_path = os.path.join(DATA_DIR, '{}.txt'.format(name))
        imgs_dir = os.path.join(DATA_DIR, name)
        mkdir_if_not_exist(imgs_dir)

        data_lines = read_file(sheet_path)
        for idx, data_line in enumerate(data_lines):
            _, img_gray = download_url_img(data_line)
            out_path = os.path.join(imgs_dir, "{}.jpg".format(str(idx).zfill(6)))
            cv2.imwrite(out_path, img_gray)
            if idx % 20 == 0:
                print('[Info] idx: {}'.format(idx))

    def extract_en_word(self, img_bgr, bboxes):
        print('[Info] img shape: {}'.format(img_bgr.shape))
        print('[Info] bboxes: {}'.format(len(bboxes)))
        draw_box_list(img_bgr, bboxes, is_show=True)
        img_patches = []
        for bbox in bboxes:
            bbox = [bbox[0], bbox[1] + 22, bbox[2], bbox[3] + 22]
            img_patch = get_cropped_patch(img_bgr, bbox)
            img_patches.append(img_patch)
        show_img_bgr(img_patches[1])

    def process(self):
        paths_list, names_list = traverse_dir_files(self.sheet_7_english)
        bboxes = self.template_boxes_dict["grade_7_sheet"]
        # img_bgr = cv2.imread(paths_list[1196])
        img_path = paths_list[1196]
        # img_path = os.path.join(DATA_DIR, 'en_sheet_templates', 'grade_7_sheet.jpg')
        img_bgr = cv2.imread(img_path)
        self.extract_en_word(img_bgr, bboxes)


def main():
    ewp = EnglishWordsParser()
    ewp.process()


if __name__ == '__main__':
    main()
