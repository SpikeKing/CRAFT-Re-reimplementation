#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.5.21
"""

import math

import pyclipper
from shapely.geometry import Polygon

from myutils.cv_utils import *


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


def boxes_from_bitmap(pred, box_thresh, min_size_ratio, aspect_ratio, unclip_ratio, min_size, binary_thresh):
    """
    使用特征图生成最终离散的检测结果
    param pred: 特征图
    param box_thresh: BBox得分阈值
    param min_size: BBox最小尺寸
    param unclip_ratio: BBox放大比例
    """
    print("[Info] binary_thresh: {}".format(binary_thresh))
    print("[Info] min_size_ratio: {}".format(min_size_ratio))
    # 生成分割结果
    _, bitmap = cv2.threshold((pred * 255).astype(np.uint8), binary_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, hierarchy=None)[-2:]

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

    print('[Info] 框数量: {}'.format(len(res_boxes)))
    return res_boxes, res_scores, res_angles


def heatmap2box(heatmap):
    """
    heatmap的值从0~1
    """
    heatmap = np.clip(heatmap, 0, 1)
    box_thresh = 0.4
    # box_thresh = 0.001
    min_size_ratio = 0.001
    aspect_ratio = 1
    unclip_ratio = 2.0
    min_size = 12
    binary_thresh = int(255 * 0.6)

    # 框来源于图像
    boxes, scores, angles = boxes_from_bitmap(
        heatmap, box_thresh, min_size_ratio, aspect_ratio,
        unclip_ratio, min_size, binary_thresh)

    return boxes, scores, angles


def main():
    pass


if __name__ == '__main__':
    main()
