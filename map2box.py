#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.5.21
"""

import copy
# import logging
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


# logger = logging.getLogger()


def get_box(box):
    box_0, box_1 = box[:, 0], box[:, 1]
    xmin = math.floor(box_0.min())
    xmax = math.ceil(box_0.max())
    ymin = math.floor(box_1.min())
    ymax = math.ceil(box_1.max())
    box_0 -= xmin
    box_1 -= ymin
    return xmin, xmax, ymin, ymax


def box_score_fast(bitmap, box, points):
    # 计算 box 包围的区域的平均得分
    xmin, xmax, ymin, ymax = get_box(box)
    _xmin, _xmax, _ymin, _ymax = get_box(points.copy())
    _xmin = max(_xmin, xmin)
    _xmax = min(_xmax, xmax)
    _ymin = max(_ymin, ymin)
    _ymax = min(_ymax, ymax)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    score = cv2.mean(bitmap[_ymin:_ymax + 1, _xmin:_xmax + 1],
                     mask[_ymin - ymin:_ymax - ymin + 1, _xmin - xmin:_xmax - xmin + 1])[0]
    return score, mask, xmin, ymin, xmax, ymax


def unclip(box, unclip_ratio):
    if unclip_ratio == 0:
        return [box]
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def azimuthAngle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if y2 > y1:
        angle = math.atan(dx / dy)
    elif y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    return (angle * 180 / math.pi)


def comput_quad_angle(quad):
    # 计算最小外接四边形框与水平角度夹角(-90,90)之间
    left = (quad[0] + quad[3]) / 2
    right = (quad[1] + quad[2]) / 2
    angle = azimuthAngle(left[0], left[1], right[0], right[1])
    if angle != 0:
        angle = angle - 90
    return angle


def get_mini_boxes(contour, aspect_ratio):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    bounding_box = cv2.minAreaRect(contour)
    polygon_area = bounding_box[1][0] * bounding_box[1][1]
    # 当正矩形框和最小外接四边形框的面积接近时，直接采用正矩形框
    if (rect_area - area) / (polygon_area - area + 1e-6) > aspect_ratio:
        points = sorted(tuple(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array((points[index_1], points[index_2],
                        points[index_3], points[index_4]))
        #  p0     p1
        #
        #  p3     p2
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    else:
        box = np.array(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    return box, h, polygon_area


PIXEL_MAX = 255
SAMPLE_MARGIN = 5


def sample_curve_points(poly_pts, poly_mask, h, w, sample_point_count):
    poly_left_ = np.amin(poly_pts, 0)[0]
    poly_right_ = np.amax(poly_pts, 0)[0]
    poly_width_ = poly_right_ - poly_left_  # 多边形的最大宽度

    pt_x_interval_ = int((poly_width_ - (2*SAMPLE_MARGIN)
                          ) / (sample_point_count + 1))

    ret_bottom_pts_ = []

    for i in range(sample_point_count + 2):
        x_ = poly_left_ + SAMPLE_MARGIN + i * pt_x_interval_
        if x_ >= poly_right_ or x_ >= w:
            break
        non_zeros_ = np.nonzero(poly_mask[:, x_])
        if non_zeros_[0].size == 0:
            break
        y_bottom_ = np.amax(non_zeros_)
        ret_bottom_pts_.append([x_, y_bottom_])

    return ret_bottom_pts_


def get_curly_text_rect(src_img, output):
    src_point0 = [int(output[0][0]), int(output[0][1])]
    src_point1 = [int(output[1][0]), int(output[1][1])]
    src_point2 = [int(output[2][0]), int(output[2][1])]
    dst_point0 = [0, 0]
    dst_point1 = [int(output[1][0]) - int(output[0][0]), 0]
    dst_point2 = [int(output[2][0]) - int(output[3][0]),
                  int(output[2][1]) - int(output[1][1])]
    cols = int(int(output[1][0]) - int(output[0][0]))
    rows = max((int(output[3][1]) - int(output[0][1])),
               (int(output[2][1]) - int(output[1][1])))

    pts1 = np.float32([src_point0, src_point1, src_point2])
    pts2 = np.float32([dst_point0, dst_point1, dst_point2])
    M_affine = cv2.getAffineTransform(pts1, pts2)

    ret = cv2.warpAffine(src_img, M_affine, (cols, rows))
    return ret


def is_curly_text(poly_pts, h, w, out_box_region_size_, quad):
    poly_left_ = np.amin(poly_pts, 0)[0]
    poly_right_ = np.amax(poly_pts, 0)[0]

    poly_top_ = np.amin(poly_pts, 0)[1]
    poly_bottom_ = np.amax(poly_pts, 0)[1]
    poly_height_ = abs(poly_bottom_ - poly_top_)

    poly_width_ = poly_right_ - poly_left_  # 多边形的最大宽度
    poly_width_ratio_ = poly_width_ / float(w)

    if poly_width_ratio_ <= 0.3 or poly_height_ < 15:
        return False, [], None, None

    poly_mask_ = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly_mask_, [poly_pts], 1)

    poly_mask_small_ = get_curly_text_rect(poly_mask_, quad)
    poly_region_size_ = np.sum(poly_mask_small_, dtype=np.uint32)  # 多边形的面积

    region_box_ratio_ = poly_region_size_ / \
        float(out_box_region_size_)  # 多边形面积占最小外接矩阵的比例

    if region_box_ratio_ >= 0.7:
        return False, [], None, None

    poly_region_heights_ = np.sum(poly_mask_small_, 0, np.uint32)
    poly_region_heights_ = poly_region_heights_[
        np.nonzero(poly_region_heights_)]
    poly_height_std_ = np.std(poly_region_heights_)
    poly_height_mean_ = np.max(poly_region_heights_)
    if (poly_height_std_ >= 4.0 or poly_height_mean_ < 15) and region_box_ratio_ >= 0.6:
        return False, [], None, None

    bottom_pts_ = sample_curve_points(poly_pts, poly_mask_, h, w, 12)
    return True, bottom_pts_, poly_height_mean_, poly_mask_


def boxes_from_bitmap(pred, box_thresh, min_size_ratio, aspect_ratio, unclip_ratio, min_size, binary_thresh):
    """
    使用特征图生成最终离散的检测结果
    param pred: 特征图
    param box_thresh: BBox得分阈值
    param min_size: BBox最小尺寸
    param unclip_ratio: BBox放大比例
    """

    # 生成分割结果
    _, bitmap = cv2.threshold((pred * 255).astype(np.uint8), binary_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, hierarchy=None)[-2:]
    ret = []

    height, width = pred.shape

    if not contours:
        return [], [], []

    res_boxes, res_scores, res_angles = [], [], []
    for contour in contours:
        points = contour.reshape((-1, 2))
        if points.shape[0] < 4:
            continue

        boxes = unclip(points, unclip_ratio)
        for box in boxes:
            box = np.array(box)
            box[:, 0] = np.clip(box[:, 0], 0, width - 1)
            box[:, 1] = np.clip(box[:, 1], 0, height - 1)
            poly_pts_ = copy.deepcopy(box)

            quad, sside, out_box_region_size_ = get_mini_boxes(box.reshape((-1, 1, 2)), aspect_ratio)
            if sside < max(min_size_ratio * (height + width), min_size):
                continue

            score, mask, x1, y1, x2, y2 = box_score_fast(pred, box, points)
            if score < box_thresh:
                continue

            angle = comput_quad_angle(quad)
            res_boxes.append([x1, y1, x2, y2])
            res_scores.append(float(score))
            res_angles.append(int(angle))

    return res_boxes, res_scores, res_angles


class Map2Box(object):
    def __init__(self):
        parameter = dict()
        self.box_thresh = float(parameter.get('box_thresh', 0.3))
        self.min_size_ratio = float(parameter.get('min_size_ratio', 0.015))
        self.unclip_ratio = float(parameter.get('unclip_ratio', 2.0))
        self.min_size = int(parameter.get('min_size', 12))
        self.pool_size = int(parameter.get('pool_size', 0))
        self.chunk_size = int(parameter.get('chunk_size', 2))
        self.binary_thresh = int(parameter.get('binary_thresh', 145))
        self.aspect_ratio = float(parameter.get('aspect_ratio', 1))
        self.check_curly_text = bool(int(parameter.get('check_curly_text', 0)))

        self.pool = ThreadPoolExecutor(self.pool_size) if self.pool_size > 0 else None

    def process_fields(self, prob_arr):
        prob_arr = np.clip(prob_arr, 0, 1)

        ret = boxes_from_bitmap(prob_arr, self.box_thresh, self.min_size_ratio, self.aspect_ratio,
                                self.unclip_ratio, self.min_size, self.binary_thresh)
        return ret


def main():
    pass


if __name__ == '__main__':
    main()
