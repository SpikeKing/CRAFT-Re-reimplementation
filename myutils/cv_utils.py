#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/13
"""

import cv2
import copy
import numpy as np


def draw_line_len(img_bgr, start_p, v_length, v_arrow, is_new=True, is_show=False, save_name=None):
    """
    绘制直线
    """
    import cv2
    import copy
    import numpy as np

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    x2 = int(start_p[0] - v_length * np.cos(v_arrow / 360 * 2 * np.pi))
    y2 = int(start_p[1] - v_length * np.sin(v_arrow / 360 * 2 * np.pi))

    cv2.arrowedLine(img_bgr, tuple(start_p), (x2, y2), color=(0, 0, 255), thickness=4, tipLength=0.4)

    if is_show:
        show_img_bgr(img_bgr, save_name=save_name)  # 显示眼睛


def draw_text(img_bgr, text, org=(3, 20), color=(0, 0, 255), scale_x=1, thickness_x=1):
    """
    绘制文字，自动调整文字大小
    """
    import cv2
    h, w, _ = img_bgr.shape
    m = h * w
    text = str(text)

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = m / 8000000
    font_scale = max(font_scale, 0.5)
    font_scale *= scale_x

    thickness = m // 4000000
    thickness = max(thickness, 1)
    thickness *= thickness_x
    # print('[Info] font_scale: {}, thickness: {}, max: {}, x: {}'.format(font_scale, thickness, m, h*w))
    lineType = 2

    img_bgr = cv2.putText(img_bgr, text, org, font, font_scale, color, thickness, lineType)
    return img_bgr


def draw_eyes(img_bgr, eyes_landmarks, radius, offsets_list, is_new=True, is_show=False, save_name=None):
    """
    绘制图像
    """
    import cv2
    import copy
    import numpy as np

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    th = 1
    eye_upscale = 1
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=eye_upscale, fy=eye_upscale)

    for el, r, offsets in zip(eyes_landmarks, radius, offsets_list):
        start_x, start_y, offset_scale = offsets
        real_el = el * offset_scale + [start_x, start_y]
        real_radius = r * offset_scale

        # 眼睛
        cv2.polylines(
            img_bgr,
            [np.round(eye_upscale * real_el[0:8]).astype(np.int32)
                 .reshape(-1, 1, 2)],
            isClosed=True, color=(255, 255, 0),
            thickness=th, lineType=cv2.LINE_AA,
        )

        # 眼球
        cv2.polylines(
            img_bgr,
            [np.round(eye_upscale * real_el[8:16]).astype(np.int32)
                 .reshape(-1, 1, 2)],
            isClosed=True, color=(0, 255, 255),
            thickness=th, lineType=cv2.LINE_AA,
        )

        iris_center = real_el[16]
        eye_center = real_el[17]

        eye_center = (real_el[0] + real_el[4]) / 2

        # 虹膜中心
        cv2.drawMarker(
            img_bgr,
            tuple(np.round(eye_upscale * iris_center).astype(np.int32)),
            color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
            thickness=th + 1, line_type=cv2.LINE_AA,
        )

        # 眼睑中心
        cv2.drawMarker(
            img_bgr,
            tuple(np.round(eye_upscale * eye_center).astype(np.int32)),
            color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
            thickness=th + 1, line_type=cv2.LINE_AA,
        )

        cv2.circle(img_bgr, center=tuple(eye_center), radius=real_radius, color=(0, 0, 255))

        if is_show:
            show_img_bgr(img_bgr, save_name=save_name)  # 显示眼睛

    return img_bgr


def draw_box(img_bgr, box, color=(0, 0, 255), is_show=True, is_new=True, tk=None, save_name=None):
    """
    绘制box
    """
    import cv2
    import copy

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    # print(x_min, y_min, x_max, y_max)

    ih, iw, _ = img_bgr.shape
    if not tk:
        m = ih * iw
        tk = m // 4000000
        tk = max(tk, 1)
    else:
        tk = tk

    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), color, tk)

    if is_show:
        show_img_bgr(img_bgr, save_name)

    return img_bgr


def draw_4p_rec(img_bgr, rec, color=(0, 0, 255), is_show=True, is_new=True):
    """
    绘制box
    """
    import copy
    import matplotlib.pyplot as plt

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    ih, iw, _ = img_bgr.shape
    # color = (0, 0, 255)
    tk = max(min(ih, iw) // 200, 2)

    rec_arr = np.array(rec)
    cv2.fillPoly(img_bgr, [rec_arr], color)

    if is_show:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()

    return img_bgr


def draw_points(img_bgr, points, is_new=True, save_name=None):
    """
    绘制多个点
    """
    import cv2
    import copy
    import matplotlib.pyplot as plt

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    color = (0, 255, 0)
    ih, iw, _ = img_bgr.shape
    r = max(min(ih, iw) // 200, 1)
    tk = -1
    for p in points:
        p = (int(p[0]), int(p[1]))
        cv2.circle(img_bgr, tuple(p), r, color, tk)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_rgb)


def draw_pie(labels, sizes):
    """
    绘制饼状图, 测试
    labels = [u'大型', u'中型', u'小型', u'微型']  # 定义标签
    sizes = [46, 253, 321, 66]  # 每块值

    :param labels: 标签
    :param sizes: 类别值
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.figure(figsize=(9, 9))  # 调节图形大小

    colors = ['yellow', 'darkorange', 'limegreen', 'lightskyblue', 'blueviolet']  # 每块颜色定义
    colors = colors[:len(labels)]
    explode = tuple([0] * len(labels))  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离

    # 设置饼图文字大小
    [t.set_size(20) for t in text1]
    [t.set_size(20) for t in text2]

    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部的文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.show()


def point2box(point, radius):
    """
    点到矩形
    :param point: 点
    :param radius: 半径
    :return: [x_min, y_min, x_max, y_max]
    """
    start_p = [point[0] - radius, point[1] - radius]
    end_p = [point[0] + radius, point[1] + radius]

    return [int(start_p[0]), int(start_p[1]), int(end_p[0]), int(end_p[1])]


def get_mask_box(mask):
    """
    mask的边框
    """
    import numpy as np
    y, x = np.where(mask)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    box = [x_min, y_min, x_max, y_max]
    return box


def get_box_size(box):
    """
    矩形尺寸
    """
    x_min, y_min, x_max, y_max = [b for b in box]
    return (x_max - x_min) * (y_max - y_min)


def get_polygon_size(box):
    """
    四边形尺寸
    """
    import cv2
    import numpy as np
    contour = np.array(box, dtype=np.int32)
    area = cv2.contourArea(contour)
    return area


def get_patch(img, box):
    """
    获取Img的Patch
    :param img: 图像
    :param box: [x_min, y_min, x_max, y_max]
    :return 图像块
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0]))
    y_min = int(max(0, box[1]))
    x_max = int(min(box[2], w))
    y_max = int(min(box[3], h))

    img_patch = img[y_min:y_max, x_min:x_max, :]
    return img_patch


def expand_patch(img, box, x):
    """
    box扩充x像素
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0] - x))
    y_min = int(max(0, box[1] - x))
    x_max = int(min(box[2] + x, w))
    y_max = int(min(box[3] + x, h))

    img_patch = img[y_min:y_max, x_min:x_max, :]
    return img_patch


def expand_box(img, box, x):
    """
    box扩充x像素
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0] - x))
    y_min = int(max(0, box[1] - x))
    x_max = int(min(box[2] + x, w))
    y_max = int(min(box[3] + x, h))

    return [x_min, y_min, x_max, y_max]


def merge_boxes(box_list):
    """
    合并多个Box
    """
    x_list, y_list = [], []

    for box in box_list:
        x_min, y_min, x_max, y_max = box
        x_list.append(x_min)
        x_list.append(x_max)
        y_list.append(y_min)
        y_list.append(y_max)
    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)

    large_box = [x_min, y_min, x_max, y_max]
    return large_box


def merge_two_box(box_a, box_b):
    """
    合并两个box
    """
    x1_min, y1_min, x1_max, y1_max = box_a
    x2_min, y2_min, x2_max, y2_max = box_b
    nx_min, ny_min = min(x1_min, x2_min), min(y1_min, y2_min)
    nx_max, ny_max = max(x1_max, x2_max), max(y1_max, y2_max)
    tmp_box = [nx_min, ny_min, nx_max, ny_max]

    return tmp_box


def min_iou(box_a, box_b):
    """
    最小框的面积占比
    """
    box_a = [int(x) for x in box_a]
    box_b = [int(x) for x in box_b]

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / min(box_a_area, box_b_area)

    return iou


def mid_point(p1, p2):
    """
    计算中点
    """
    x = (p1[0] + p2[0]) // 2
    y = (p1[1] + p2[1]) // 2
    return [x, y]


def generate_colors(n_colors, seed=47):
    """
    随机生成颜色
    """
    import numpy as np

    np.random.seed(seed)
    color_list = []
    for i in range(n_colors):
        color = (np.random.random((1, 3)) * 0.8).tolist()[0]
        color = [int(j * 255) for j in color]
        # color = list(np.clip(color, 0, 255))
        color_list.append(color)

    return color_list


def show_img_bgr(img_bgr, save_name=None):
    """
    展示BGR彩色图
    """
    import cv2
    import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_rgb)


def show_img_gray(img_gray, save_name=None):
    """
    展示灰度图
    """
    import matplotlib.pyplot as plt

    plt.imshow(img_gray)
    plt.show()
    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_gray)


def init_vid(vid_path):
    """
    初始化视频
    """
    import cv2

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

    return cap, n_frame, fps, h, w


def unify_size(h, w, ms):
    """
    统一最长边的尺寸

    :h 高
    :w 宽
    :ms 最长尺寸
    """
    # 最长边修改为标准尺寸
    if w > h:
        r = ms / w
    else:
        r = ms / h
    h = int(h * r)
    w = int(w * r)

    return h, w


def get_fixes_frames(n_frame, max_gap):
    """
    等比例抽帧

    :param n_frame: 总帧数
    :param max_gap: 抽帧数量
    :return: 帧索引
    """
    from math import floor

    idx_list = []
    if n_frame > max_gap:
        v_gap = float(n_frame) / float(max_gap)  # 只使用100帧
        for gap_idx in range(max_gap):
            idx = int(floor(gap_idx * v_gap))
            idx_list.append(idx)
    else:
        for gap_idx in range(n_frame):
            idx_list.append(gap_idx)
    return idx_list


def sigmoid_thr(val, thr, gap, reverse=False):
    """
    数值归一化

    thr: 均值
    gap: 区间，4~5等分
    """
    import numpy as np
    x = val - thr
    if reverse:
        x *= -1
    x = x / gap
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 4)  # 保留4位


def write_video(vid_path, frames, fps, h, w):
    """
    写入视频
    :param vid_path: 输入视频的URL
    :param frames: 帧列表
    :param fps: FPS
    :param w: 视频宽
    :param h: 视频高
    :return: 写入完成的视频路径
    """
    import cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以
    vw = cv2.VideoWriter(filename=vid_path, fourcc=fourcc, fps=fps, frameSize=(w, h), isColor=True)

    for frame in frames:
        vw.write(frame)

    vw.release()
    return vid_path


def merge_imgs(imgs, cols=6, rows=6, is_h=True):
    """
    合并图像
    :param imgs: 图像序列
    :param cols: 行数
    :param rows: 列数
    :param is_h: 是否水平排列
    :param sk: 间隔，当sk=2时，即0, 2, 4, 6
    :return: 大图
    """
    import numpy as np

    if not imgs:
        raise Exception('[Exception] 合并图像的输入为空!')

    img_shape = imgs[0].shape
    h, w, _ = img_shape

    large_imgs = np.ones((rows * h, cols * w, 3)) * 255  # 大图

    if is_h:
        for j in range(rows):
            for i in range(cols):
                idx = j * cols + i
                if idx > len(imgs) - 1:  # 少于帧数，输出透明帧
                    break
                # print('[Info] 帧的idx: {}, i: {}, j:{}'.format(idx, i, j))
                large_imgs[(j * h):(j * h + h), (i * w): (i * w + w)] = imgs[idx]
                # print(large_imgs.shape)
                # show_png(large_imgs)
        # show_png(large_imgs)
    else:
        for i in range(cols):
            for j in range(rows):
                idx = i * cols + j
                if idx > len(imgs) - 1:  # 少于帧数，输出透明帧
                    break
                large_imgs[(j * h):(j * h + h), (i * w): (i * w + w)] = imgs[idx]

    return large_imgs


def merge_two_imgs(img1, img2):
    """
    左右合并2张图像, 高度相同, 宽度等比例变化
    """
    import cv2
    import numpy as np

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h = min(h1, h2)
    n_w1 = int(w1 * h / h1)
    n_w2 = int(w2 * h / h2)
    n_img1 = cv2.resize(img1, (n_w1, h))
    n_img2 = cv2.resize(img2, (n_w2, h))

    large_img = np.ones((h, n_w1 + n_w2, 3)) * 255
    large_img[:, 0: n_w1] = n_img1
    large_img[:, n_w1: n_w1+n_w2] = n_img2
    large_img = large_img.astype(np.uint8)

    return large_img


def rotate_img_with_bound(img_np, angle):
    """
    旋转图像角度
    注意angle是顺时针还是逆时针
    """
    import cv2
    import numpy as np

    angle *= -1
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img_np.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # !!!注意angle是顺时针还是逆时针
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    img_rotated = cv2.warpAffine(img_np, M, (nW, nH))
    return img_rotated, M


def resize_img_fixed(img, x, is_height=True):
    """
    resize图像，根据某一个边的长度
    """
    import cv2

    h, w, _ = img.shape
    if is_height:
        nh = x
        nw = int(w * nh / h)
    else:
        nw = x
        nh = int(h * nw / w)

    img_r = cv2.resize(img, (nw, nh))
    return img_r


def random_crop(img, height, width, sh=0, sw=0):
    """
    随机剪裁
    """
    import random
    # print(sh, img.shape[0] - height - sh)
    h, w, _ = img.shape
    img = img[sh:h-sh, sw:w-sw]
    h, w, _ = img.shape

    y = random.randint(0, h - height)
    x = random.randint(0, w - width)

    img = img[y:y+height, x:x+width]
    return img


def format_angle(angle):
    """
    格式化角度
    """
    angle = int(angle)
    if angle <= 45 or angle >= 325:
        r_angle = 0
    elif 45 < angle <= 135:
        r_angle = 90
    elif 135 < angle <= 225:
        r_angle = 180
    else:
        r_angle = 270
    return r_angle


def resize_with_padding(img_bgr, size, padding_bgr=None):
    """
    先放大，再resize图像(方图)
    """
    h, w, c = img_bgr.shape
    if h > w:
        new_h = size
        new_w = int(size / h * w)
    else:
        new_h = int(size / w * h)
        new_w = size
    img_resize = cv2.resize(img_bgr, (new_w, new_h))
    img_new = np.ones((size, size, 3), dtype=np.uint8) * 255

    if padding_bgr:  # 设置颜色
        color = tuple(padding_bgr)
        img_new[:] = color

    sh = (size - new_h) // 2
    sw = (size - new_w) // 2
    img_new[sh:new_h + sh, sw:new_w + sw] = img_resize
    return img_new


def rotate_img_for_4angle(img_bgr, angle):
    """
    旋转4个角度
    """
    # print('[Info] angle: {}'.format(angle))
    angle = int(angle)
    if angle not in [0, 90, 180, 270]:
        raise Exception('[Exception] angle not in [0, 90, 180, 270]')
    import cv2
    if angle == 90:
        img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle == 270:
        img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        img_rotated = img_bgr
    return img_rotated


def rec2box(rec):
    """
    多边形(多点)转换为xyxy
    """
    x_list, y_list = [], []
    for pnt in rec:
        x_list.append(pnt[0])
        y_list.append(pnt[1])
    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)
    box = [x_min, y_min, x_max, y_max]
    return box


def get_box_center(box):
    """
    获取bbox的中心
    """
    x_min, y_min, x_max, y_max = box
    x = (x_min + x_max) // 2
    y = (y_min + y_max) // 2
    return x, y


def get_rec_center(rec):
    """
    获取四点矩形的中心
    """
    x_list, y_list = [], []
    for pnt in rec:
        x_list.append(pnt[0])
        y_list.append(pnt[1])
    x_min, y_min, x_max, y_max = min(x_list), min(y_list), max(x_list), max(y_list)
    x = (x_min + x_max) // 2
    y = (y_min + y_max) // 2
    return x, y


def draw_box_list(img_bgr, box_list, is_arrow=False, is_text=True, is_show=False, is_new=False, save_name=None):
    """
    绘制矩形列表
    """
    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    n_box = len(box_list)
    color_list = generate_colors(n_box)  # 随机生成颜色
    ori_img = copy.copy(img_bgr)
    img_copy = copy.copy(img_bgr)

    # 绘制颜色块
    for idx, (box, color) in enumerate(zip(box_list, color_list)):
        # rec_arr = np.array(box)
        # ori_img = cv2.fillPoly(ori_img, [rec_arr], color_list[idx])
        x_min, y_min, x_max, y_max = box
        ori_img = cv2.rectangle(ori_img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(color), thickness=-1)

    ori_img = cv2.addWeighted(ori_img, 0.5, img_copy, 0.5, 0)
    ori_img = np.clip(ori_img, 0, 255)

    # 绘制方向和序号
    pre_point, next_point = None, None
    pre_color = None
    for idx, box in enumerate(box_list):
        x_min, y_min, x_max, y_max = box
        point = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        if is_arrow:
            pre_point = point
            if pre_point and next_point:  # 绘制箭头
                cv2.arrowedLine(ori_img, next_point, pre_point, pre_color, thickness=5,
                                line_type=cv2.LINE_4, shift=0, tipLength=0.05)
            next_point = point
            pre_color = color_list[idx]
        if is_text:
            draw_text(ori_img, str(idx), point)  # 绘制序号

    if is_show or save_name:
        show_img_bgr(ori_img, save_name=save_name)
    return ori_img


def draw_rec_list(img_bgr, rec_list, is_text=True, is_show=False, is_new=False, save_name=None):
    """
    绘制4点的四边形
    """
    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    n_rec = len(rec_list)
    color_list = generate_colors(n_rec)  # 随机生成颜色
    ori_img = copy.copy(img_bgr)
    img_copy = copy.copy(img_bgr)

    # 绘制颜色块
    for idx, (rec, color) in enumerate(zip(rec_list, color_list)):
        rec_arr = np.array(rec)
        ori_img = cv2.fillPoly(ori_img, [rec_arr], color_list[idx])
        # x_min, y_min, x_max, y_max = box
        # ori_img = cv2.rectangle(ori_img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(color), thickness=-1)

    ori_img = cv2.addWeighted(ori_img, 0.5, img_copy, 0.5, 0)
    ori_img = np.clip(ori_img, 0, 255)

    # 绘制方向和序号
    for idx, rec in enumerate(rec_list):
        point = get_rec_center(rec)
        if is_text:
            draw_text(ori_img, str(idx), point)  # 绘制序号

    if is_show or save_name:
        show_img_bgr(ori_img, save_name=save_name)
    return ori_img



def safe_div(x, y):
    """
    安全除法
    :param x: 分子
    :param y: 分母
    :return: 除法
    """
    x = float(x)
    y = float(y)
    if y == 0.0:
        return 0.0
    return x / y


def check_line_intersect(line1, line2, thr=0.33):
    """
    检测连线是不是交叉
    """
    line1_x, line2_x = sorted((line1, line2))
    diff = line1_x[1] - line2_x[0]
    r = safe_div(diff, min(line1_x[1] - line1_x[0], line2_x[1] - line2_x[0]))
    if r > thr:
        return True, r
    else:
        return False, r


def sorted_boxes_by_col(boxes, img_bgr=None):
    """
    根据位置, 按列排序boxes
    """
    if len(boxes) == 1:
        return [boxes], [[0]], 1

    x_min_list, y_min_list, s_min_list = [], [], []

    # 从左到右(lr)、从上到下(ud)排序
    for box in boxes:
        x_min_list.append((box[0] + box[2]) // 2)
        y_min_list.append((box[1] + box[3]) // 2)

    box_ud_idxes = np.argsort(y_min_list)
    box_lr_idxes = np.argsort(x_min_list)
    sorted_boxes, sorted_idxes = [], []  # 最终的box结果
    num_row = 0

    n_boxes = len(boxes)
    idx_flag = [False] * len(boxes)

    for i in range(n_boxes):
        line_boxes = list()
        line_idxes = list()

        box_idx = box_ud_idxes[i]
        if idx_flag[box_idx]:
            continue
        idx_flag[box_idx] = True

        target_box = boxes[box_idx]

        target_height = [target_box[1], target_box[3]]
        target_width = [target_box[0], target_box[2]]

        lr_idx = np.where(box_lr_idxes == box_idx)[0]
        lr_idx = int(lr_idx)

        line_boxes.append(target_box)
        line_idxes.append(box_idx)

        for l_i in range(lr_idx - 1, -1, -1):  # 从当前框，向上查找
            tmp_box_idx = box_lr_idxes[l_i]
            if idx_flag[tmp_box_idx]:
                continue
            tmp_box = boxes[tmp_box_idx]

            tmp_height = [tmp_box[1], tmp_box[3]]
            tmp_width = [tmp_box[0], tmp_box[2]]

            is_height_intersect, r_height = check_line_intersect(target_height, tmp_height)
            is_width_intersect, r_width = check_line_intersect(target_width, tmp_width)

            if is_width_intersect and r_height < 0.6:
                idx_flag[tmp_box_idx] = True
                if r_width < 1:
                    target_height = [tmp_box[1], tmp_box[3]]
                line_boxes.append(tmp_box)
                line_idxes.append(tmp_box_idx)

        line_boxes.reverse()
        line_idxes.reverse()

        target_height = [target_box[1], target_box[3]]
        target_width = [target_box[0], target_box[2]]

        for r_i in range(lr_idx + 1, len(box_lr_idxes)):
            tmp_box_idx = box_lr_idxes[r_i]
            if idx_flag[tmp_box_idx]:
                continue
            tmp_box = boxes[tmp_box_idx]
            tmp_height = [tmp_box[1], tmp_box[3]]
            tmp_width = [tmp_box[0], tmp_box[2]]

            is_height_intersect, r_height = check_line_intersect(target_height, tmp_height)
            is_width_intersect, r_width = check_line_intersect(target_width, tmp_width)

            if is_width_intersect and (r_height < 0.6 or r_width > 0.8):
                idx_flag[tmp_box_idx] = True
                if r_width < 1:
                    target_height = [tmp_box[1], tmp_box[3]]
                line_boxes.append(tmp_box)
                line_idxes.append(tmp_box_idx)

        y_list = [box[1] for box in line_boxes]
        line_tuple = zip(y_list, line_boxes, line_idxes)
        sorted_tuple = sorted(line_tuple)  # 排序idxes
        y_list, line_boxes, line_idxes = zip(*sorted_tuple)

        sorted_boxes.append(list(line_boxes))
        sorted_idxes.append(list(line_idxes))
        num_row += 1

    return sorted_boxes, sorted_idxes, num_row


def sorted_boxes_by_row(boxes, img_bgr=None):
    """
    根据位置, 按行排序boxes
    """
    if len(boxes) == 1:
        return [boxes], [[0]], 1

    x_min_list, y_min_list, s_min_list = [], [], []
    n_boxes = len(boxes)
    idx_flag = [False] * len(boxes)

    # 从左到右(lr)、从上到下(ud)排序
    for box in boxes:
        x_min_list.append((box[0] + box[2]) // 2)
        y_min_list.append((box[1] + box[3]) // 2)

    box_ud_idxes = np.argsort(y_min_list)
    box_lr_idxes = np.argsort(x_min_list)

    sorted_boxes, sorted_idxes = [], []  # 最终的box结果

    num_row = 0

    for i in range(n_boxes):
        line_boxes = list()
        line_idxes = list()

        box_idx = box_ud_idxes[i]
        if idx_flag[box_idx]:
            continue
        idx_flag[box_idx] = True

        target_box = boxes[box_idx]

        target_height = [target_box[1], target_box[3]]
        target_width = [target_box[0], target_box[2]]

        ud_idx = np.where(box_lr_idxes == box_idx)[0]
        ud_idx = int(ud_idx)

        line_boxes.append(target_box)
        line_idxes.append(box_idx)

        for l_i in range(ud_idx - 1, -1, -1):
            tmp_box_idx = box_lr_idxes[l_i]
            if idx_flag[tmp_box_idx]:
                continue
            tmp_box = boxes[tmp_box_idx]
            tmp_height = [tmp_box[1], tmp_box[3]]
            tmp_width = [tmp_box[0], tmp_box[2]]

            is_height_intersect, r_height = check_line_intersect(target_height, tmp_height)
            is_width_intersect, r_width = check_line_intersect(target_width, tmp_width)

            if is_height_intersect and r_width < 0.6:
                idx_flag[tmp_box_idx] = True
                if r_height < 1:
                    target_height = [tmp_box[1], tmp_box[3]]
                line_boxes.append(tmp_box)
                line_idxes.append(tmp_box_idx)

        line_boxes.reverse()
        line_idxes.reverse()

        target_height = [target_box[1], target_box[3]]
        target_width = [target_box[0], target_box[2]]

        for r_i in range(ud_idx + 1, len(box_lr_idxes)):
            tmp_box_idx = box_lr_idxes[r_i]
            if idx_flag[tmp_box_idx]:
                continue
            tmp_box = boxes[tmp_box_idx]
            tmp_height = [tmp_box[1], tmp_box[3]]
            tmp_width = [tmp_box[0], tmp_box[2]]

            is_height_intersect, r_height = check_line_intersect(target_height, tmp_height)
            is_width_intersect, r_width = check_line_intersect(target_width, tmp_width)

            if is_height_intersect and r_width < 0.6:
                idx_flag[tmp_box_idx] = True
                if r_height < 1:
                    target_height = [tmp_box[1], tmp_box[3]]
                line_boxes.append(tmp_box)
                line_idxes.append(tmp_box_idx)

        x_list = [box[0] for box in line_boxes]
        line_tuple = zip(x_list, line_boxes, line_idxes)
        sorted_tuple = sorted(line_tuple)  # 排序idxes
        x_list, line_boxes, line_idxes = zip(*sorted_tuple)

        sorted_boxes.append(list(line_boxes))
        sorted_idxes.append(list(line_idxes))
        num_row += 1

    return sorted_boxes, sorted_idxes, num_row


def filer_boxes_by_size(boxes, r_thr=0.4):
    """
    根据是否重叠过滤包含在内部的框
    """
    if not boxes:
        return boxes, [i for i in range(len(boxes))]

    size_list = []
    idx_list = []
    for idx, box in enumerate(boxes):
        size_list.append(get_box_size(box))
        idx_list.append(idx)

    def sort_three_list(list1, list2, list3, reverse=False):
        list1, list2, list3 = (list(t) for t in zip(*sorted(zip(list1, list2, list3), reverse=reverse)))
        return list1, list2, list3

    size_list, sorted_idxes, sorted_boxes = \
        sort_three_list(size_list, idx_list, boxes, reverse=True)

    n_box = len(sorted_boxes)  # box的数量
    flag_list = [True] * n_box

    for i in range(n_box):
        if not flag_list[i]:
            continue
        x_boxes = [sorted_boxes[i]]
        for j in range(i + 1, n_box):
            box1 = sorted_boxes[i]
            box2 = sorted_boxes[j]
            r_iou = min_iou(box1, box2)
            if r_iou > r_thr:
                flag_list[j] = False
                x_boxes.append(box2)
        sorted_boxes[i] = merge_boxes(x_boxes)

    new_boxes, new_idxes = [], []
    for i in range(n_box):
        if flag_list[i]:
            new_boxes.append(sorted_boxes[i])
            new_idxes.append(sorted_idxes[i])

    return new_boxes, new_idxes


def main():
    import os
    import cv2
    from root_dir import DATA_DIR
    img_path = os.path.join(DATA_DIR, 'error_imgs', 'error1_20201127.270.jpg')
    img_bgr = cv2.imread(img_path)
    img_bgr = rotate_img_with_bound(img_bgr, 90)
    show_img_bgr(img_bgr)


if __name__ == '__main__':
    main()