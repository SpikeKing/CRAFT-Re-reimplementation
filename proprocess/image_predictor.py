#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 10.6.21
"""

import os
import sys
import torch

from collections import OrderedDict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from craft import CRAFT
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
        # self.model_path = os.path.join(DATA_DIR, 'models', 'craft_mlt_25k.pth')
        self.model_path = os.path.join(DATA_DIR, 'models', 'craft_best_20210610.pth')
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

    def predict_img(self, img_rgb):
        print('[Info] img_rgb: {}'.format(img_rgb.shape))
        oh, ow, _ = img_rgb.shape

        show_img_bgr(img_rgb[:, :, ::-1])
        square_size = 2240
        interpolation = cv2.INTER_LINEAR
        mag_ratio = 2.0
        img_resized, target_ratio, size_heatmap = \
            self.resize_aspect_ratio(img_rgb, square_size, interpolation, mag_ratio)
        # show_img_bgr(img_resized[:, :, ::-1].astype(np.uint8))
        print("[Info] img_resized: {}".format(img_resized.shape))
        print("[Info] target_ratio: {}".format(target_ratio))
        print("[Info] size_heatmap: {}".format(size_heatmap))

        ratio_h = ratio_w = 1 / target_ratio

        x = self.normalize_mean_var(img_resized)
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

        img_mask = cv2.resize(score_text, None, fx=ratio_w * 2.0, fy=ratio_h * 2.0)  # 恢复原尺寸
        img_mask = img_mask[0:oh, 0:ow]
        img_mask = self.cvt_2_heatmap_img(img_mask)
        show_img_bgr(img_mask)
        print('[Info] img_mask: {}'.format(img_mask.shape))

        img_rgb = img_rgb.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_with_mask = cv2.addWeighted(img_bgr, 0.8, img_mask, 0.2, 0)
        img_with_mask = np.clip(img_with_mask, 0, 255).astype(np.uint8)
        show_img_bgr(img_with_mask, save_name=os.path.join(DATA_DIR, "img_with_mask.jpg"))

        heatmap2box(score_text, img_bgr)
        print('[Info] 处理完成')

    def process(self):
        print('[Info] 处理开始!')
        img_path = os.path.join(DATA_DIR, 'imgs', '0001.png')
        img_rgb = self.load_rgb_image(img_path)
        self.predict_img(img_rgb)


def main():
    ip = ImagePredictor()
    ip.process()


if __name__ == '__main__':
    main()
