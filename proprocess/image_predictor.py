#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 10.6.21
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import craft_utils

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from craft import CRAFT
from x_utils.vpf_utils import get_ocr_service_with_np
from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.heatmap2box import heatmap2box
from root_dir import DATA_DIR


class ImagePredictor(object):
    """
    预测图像
    """
    def __init__(self):
        """
        初始化
        """
        self.model_path = os.path.join(DATA_DIR, 'models', 'best_model_20210615.pth')
        # self.model_path = os.path.join(DATA_DIR, 'models', 'craft_best_20210611.pth')
        print('[Info] 模型路径: {}'.format(self.model_path))
        self.cuda = False
        print('[Info] 是否GPU: {}'.format(self.cuda))
        self.net = self.load_model(self.model_path, self.cuda)

    @staticmethod
    def copy_state_dict(state_dict):
        """
        加载数据字典
        """
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def load_model(self, model_path, is_cuda=False):
        """
        加载模型
        """
        net = CRAFT()
        if not is_cuda:
            net.load_state_dict(self.copy_state_dict(torch.load(model_path, map_location='cpu')))
        else:
            net.load_state_dict(self.copy_state_dict(torch.load(model_path)))

        if is_cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()
        return net

    @staticmethod
    def load_rgb_image(img_path):
        """
        加载RGB图像
        """
        img = cv2.imread(img_path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # 输出图像是RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def resize_aspect_ratio(img_rgb, square_size, interpolation, mag_ratio=1.0):
        """
        调整图像大小
        输入:
        img_rgb 输入图像，square_size 最大尺寸，interpolation 差值方式，mag_ratio 放大比例
        输出：
        resized： resize图像，长宽需要补全到32的倍数
        ratio：resize比例，一般是mag_ratio，尺寸超过square_size，换成其他值
        size_heatmap：热力图尺寸，是resize图像的1/2
        """
        height, width, channel = img_rgb.shape

        # magnify image size
        target_size = mag_ratio * max(height, width)

        # set original image size
        if target_size > square_size:  # 最大尺寸
            target_size = square_size

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img_rgb, (target_w, target_h), interpolation=interpolation)

        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc

        # heatmap是预测图像的一半尺寸
        target_h, target_w = target_h32, target_w32
        size_heatmap = (int(target_w / 2), int(target_h / 2))

        return resized, ratio, size_heatmap

    @staticmethod
    def normalize_mean_var(img_rgb, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """
        正则化
        """
        img = img_rgb.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0],
                        dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
                        dtype=np.float32)
        return img

    @staticmethod
    def cvt_2_heatmap_img(img):
        """
        灰度图转换为颜色图像
        """
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    @staticmethod
    def pos2bbox(pos):
        rec = []
        for point in pos:
            x = point["x"]
            y = point["y"]
            rec.append([x, y])
        bbox = rec2bbox(rec)
        return bbox

    @staticmethod
    def parse_ocr_hw_res(res_dict):
        """
        解析ocr的手写结果
        """
        data_dict = res_dict['data']['data']
        words_info = data_dict['wordsInfo']
        bbox_list, line_list, prob_list = [], [], []
        num_of_hw = 0
        for word_info in words_info:
            rec_classify = word_info['recClassify']
            pos = word_info['pos']
            if rec_classify == 25:
                bbox = ImagePredictor.pos2bbox(pos)
                word = word_info['word']
                prob = word_info['prob']
                bbox_list.append(bbox)
                line_list.append(word)
                prob_list.append(prob)
                num_of_hw += 1
        print('[Info] 手写框数量: {}'.format(num_of_hw))
        print('[Info] bbox_list: {}, word_list: {}, prob_list: {}'
              .format(len(bbox_list), len(line_list), len(prob_list)))

        return bbox_list, line_list, prob_list

    def predict_hw_boxes(self, img_rgb):
        """
        预测手写文字区域
        """
        res_dict = get_ocr_service_with_np(img_rgb)
        hw_bboxes, hw_lines, hw_probs = ImagePredictor.parse_ocr_hw_res(res_dict)

        # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # draw_box_list(img_bgr, box_list=bbox_list, is_show=True)
        return hw_bboxes, hw_lines, hw_probs

    def predict_feature_map(self, img_rgb):
        """
        预测FeatureMap
        """
        x = self.normalize_mean_var(img_rgb)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))
        print("[Info] x: {}".format(x.shape))
        if self.cuda:
            x = x.cuda()

        # forward pass
        pred_s_time = time.time()
        y, _ = self.net(x)
        pred_time = time.time() - pred_s_time
        print('[Info] y: {}, elapsed: {} ms'.format(y.shape, pred_time))

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        print('[Info] score_text: {}, score_link: {}'.format(score_text.shape, score_link.shape))
        return score_text, score_link

    def predict_en_word_bboxes(self, img_rgb):
        """
        预测英语词框
        """
        print('[Info] img_rgb: {}'.format(img_rgb.shape))
        oh, ow, _ = img_rgb.shape
        # show_img_bgr(img_rgb[:, :, ::-1])

        # 第一步
        square_size = 1120
        mag_ratio = 2.0
        interpolation = cv2.INTER_LINEAR

        image_resized, target_ratio, size_heatmap = \
            ImagePredictor.resize_aspect_ratio(img_rgb, square_size, interpolation, mag_ratio)
        # show_img_bgr(img_resized[:, :, ::-1].astype(np.uint8))
        print("[Info] img_resized: {}".format(image_resized.shape))
        print("[Info] target_ratio: {}".format(target_ratio))
        print("[Info] size_heatmap: {}".format(size_heatmap))

        score_text, score_link = self.predict_feature_map(image_resized)  # 预测特征图

        # 参数
        text_thresh = 0.4  # 默认0.7
        link_thresh = 0.4
        low_text = 0.2  # 默认0.4
        poly = False
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_thresh, link_thresh, low_text, poly)

        ratio_h = ratio_w = 1 / target_ratio
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        bboxes = []
        for i in range(len(boxes)):
            rec = boxes[i].astype(np.int)
            bboxes.append(rec2bbox(rec))

        fake_scores = [1.0 for i in range(len(boxes))]

        return bboxes, fake_scores

    def predict_char_bboxes(self, img_rgb):
        """
        预测中文汉字
        """
        print('[Info] img_rgb: {}'.format(img_rgb.shape))
        oh, ow, _ = img_rgb.shape
        # show_img_bgr(img_rgb[:, :, ::-1])

        # 第一步
        square_size = 1120
        mag_ratio = 2.0
        interpolation = cv2.INTER_LINEAR

        image_resized, target_ratio, size_heatmap = \
            ImagePredictor.resize_aspect_ratio(img_rgb, square_size, interpolation, mag_ratio)
        # show_img_bgr(img_resized[:, :, ::-1].astype(np.uint8))
        print("[Info] img_resized: {}".format(image_resized.shape))
        print("[Info] target_ratio: {}".format(target_ratio))
        print("[Info] size_heatmap: {}".format(size_heatmap))

        score_text, score_link = self.predict_feature_map(image_resized)  # 预测特征图

        # x = self.normalize_mean_var(image_resized)
        # x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        # x = Variable(x.unsqueeze(0))
        # print("[Info] x: {}".format(x.shape))
        # if self.cuda:
        #     x = x.cuda()
        #
        # # forward pass
        # pred_s_time = time.time()
        # y, _ = self.net(x)
        # pred_time = time.time() - pred_s_time
        # print('[Info] y: {}, elapsed: {} ms'.format(y.shape, pred_time))
        #
        # score_text = y[0, :, :, 0].cpu().data.numpy()
        # score_link = y[0, :, :, 1].cpu().data.numpy()
        # print('[Info] score_text: {}, score_link: {}'.format(score_text.shape, score_link.shape))

        ratio_h = ratio_w = 1 / target_ratio
        score_text = cv2.resize(score_text, None, fx=ratio_w * 2.0, fy=ratio_h * 2.0)  # 恢复原尺寸
        score_text = score_text[0:oh, 0:ow]
        img_mask = ImagePredictor.cvt_2_heatmap_img(score_text)
        show_img_bgr(img_mask)
        print('[Info] img_mask: {}'.format(img_mask.shape))

        img_rgb = img_rgb.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_with_mask = cv2.addWeighted(img_bgr, 0.8, img_mask, 0.2, 0)
        img_with_mask = np.clip(img_with_mask, 0, 255).astype(np.uint8)
        show_img_bgr(img_with_mask, save_name=os.path.join(DATA_DIR, "img_with_mask.jpg"))

        bboxes, scores, angles = heatmap2box(score_text)
        print('[Info] 处理完成')
        # draw_box_list(img_bgr, boxes, is_overlap=False, is_show=True, is_text=False,
        #               save_name=os.path.join(DATA_DIR, "img_with_boxes.jpg"))
        return bboxes, scores

    @staticmethod
    def image_to_base64(image_np, ext='.jpg'):
        """
        转换为base64, ext是编码格式，'.jpg'和'.png'都支持
        """
        import cv2
        import base64

        # image = cv2.imencode('.png', image_np)[1]
        image = cv2.imencode(ext, image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]  # 生成编码
        return image_code

    @staticmethod
    def filter_and_format_bboxes(img_rgb, char_bboxes, char_scores, hw_bboxes, hw_lines, hw_probs):
        """
        过滤字符框
        """
        hw_char_data = []
        hw_char_bboxes = []
        for hw_bbox, hw_line, hw_prob in zip(hw_bboxes, hw_lines, hw_probs):
            hw_char_list, hw_score_list = [], []
            for char_bbox, char_score in zip(char_bboxes, char_scores):
                v_iou = min_iou(char_bbox, hw_bbox)
                if v_iou > 0.5:
                    hw_char_list.append(char_bbox)
                    hw_char_list, _, _ = sorted_boxes_by_row(hw_char_list)
                    hw_char_list = unfold_nested_list(hw_char_list)
                    hw_score_list.append(char_score)
                    hw_char_bboxes.append(char_bbox)  # 整体的char框bbox
            print('[Info] hw_char_list: {}, hw_line: {} - {}'
                  .format(len(hw_char_list), len(hw_line), hw_line))

            for char_idx, (char_box, char_score) in enumerate(zip(hw_char_list, hw_score_list)):
                img_char = get_cropped_patch(img_rgb, char_box)
                content = image_to_base64(img_char)
                char_data_dict = dict()
                char_data_dict["position"] = char_box
                char_data_dict["content"] = content
                char_data_dict["content_size"] = img_char.shape
                if len(hw_char_list) == len(hw_line):
                    char_data_dict["model_output_text"] = hw_line[char_idx]
                else:
                    char_data_dict["model_output_text"] = ""
                char_data_dict["bbox_score"] = round(char_score, 2)
                char_data_dict["hw_score"] = float(hw_prob) / float(100)
                hw_char_data.append(char_data_dict)

        print('[Info] num of char bboxes: {}'.format(len(hw_char_data)))
        return hw_char_data, hw_char_bboxes

    def process(self):
        print('[Info] 处理开始!')
        # img_path = os.path.join(DATA_DIR, 'imgs', 'hwg_0000000.jpg')
        # img_path = os.path.join(DATA_DIR, 'imgs', '0001.jpg')
        # img_path = os.path.join(DATA_DIR, 'imgs', 'test1.png')
        # img_path = os.path.join(DATA_DIR, 'imgs', '190101_00_1_0.jpg')
        img_path = os.path.join(DATA_DIR, 'imgs', 'IMG_20210519_092552.jpg')
        out_dir = os.path.join(DATA_DIR, 'tmps')
        mkdir_if_not_exist(out_dir)

        print('[Info] 图像路径: {}'.format(img_path))
        img_rgb = self.load_rgb_image(img_path)
        print('[Info] img_rgb: {}'.format(img_rgb.shape))
        img_bgr = img_rgb[:, :, ::-1]

        # char_bboxes, char_scores = self.predict_char_bboxes(img_rgb)
        char_bboxes, char_scores = self.predict_en_word_bboxes(img_rgb)
        out_char_bboxes = os.path.join(out_dir, 'char_bboxes.jpg')
        draw_box_list(img_bgr, char_bboxes, color=(255, 0, 0),
                      is_text=False, is_overlap=False, save_name=out_char_bboxes)
        print('[Info] 绘制字符框: {}'.format(out_char_bboxes))

        hw_bboxes, hw_lines, hw_probs = self.predict_hw_boxes(img_rgb)
        out_hw_bboxes = os.path.join(out_dir, 'hw_bboxes.jpg')
        draw_box_list(img_bgr, hw_bboxes, color=(0, 0, 255),
                      is_overlap=False, is_text=False, save_name=out_hw_bboxes)
        print('[Info] 绘制手写框: {}'.format(out_hw_bboxes))

        hw_char_data, hw_char_bboxes = self.filter_and_format_bboxes(
            img_rgb, char_bboxes, char_scores, hw_bboxes, hw_lines, hw_probs)

        out_hw_char_bboxes = os.path.join(out_dir, 'hw_char_bboxes.jpg')
        draw_box_list(img_bgr, hw_char_bboxes, color=(0, 255, 0),
                      is_overlap=False, is_text=False, save_name=out_hw_char_bboxes)

        for idx, hw_char_bbox in enumerate(hw_char_bboxes):
            if idx == 10:
                break
            img_patch = get_cropped_patch(img_bgr, hw_char_bbox)
            img_patch = cv2.resize(img_patch, None, fx=10.0, fy=10.0)
            img_patch_path = os.path.join(out_dir, '{}.jpg'.format(idx))
            cv2.imwrite(img_patch_path, img_patch)

        print('[Info] 处理完成!')


def main():
    # ip = ImagePredictor()
    # ip.process()
    url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Character-Detection/CRAFT/tmps/IMG_20210519_092552.jpg"
    _, img = download_url_img(url)
    print(img.shape)


if __name__ == '__main__':
    main()
