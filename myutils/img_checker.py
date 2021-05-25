#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 10.11.20
"""

import os
import shutil
from multiprocessing import Pool

import cv2
import argparse


def sort_two_list(list1, list2):
    """
    排序两个列表
    :param list1: 列表1
    :param list2: 列表2
    :return: 排序后的两个列表
    """
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def traverse_dir_files(root_dir, ext=None):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param ext: 后缀名
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    if not names_list:  # 文件夹为空
        return paths_list, names_list
    paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def check_img(path, size):
    is_good = True
    try:
        img_bgr = cv2.imread(path)
        h, w, _ = img_bgr.shape
        if h < size or w < size:
            is_good = False
        img_re = cv2.resize(img_bgr, (224, 224))
    except Exception as e:
        is_good = False

    if not is_good:
        print('[Info] error path: {}'.format(path))
        os.remove(path)
    # else:
    #     print('[Info] path: {}'.format(path))


def check_error(img_dir, n_prc, size):
    """
    检查错误图像的数量
    """
    print('[Info] 处理文件夹路径: {}'.format(img_dir))
    paths_list, names_list = traverse_dir_files(img_dir)
    print('[Info] 数据总量: {}'.format(len(paths_list)))

    pool = Pool(processes=n_prc)  # 多线程下载
    for idx, path in enumerate(paths_list):
        # check_img(path, size)
        pool.apply_async(check_img, (path, size))
        if (idx+1) % 1000 == 0:
            print('[Info] idx: {}'.format(idx+1))

    pool.close()
    pool.join()

    print('[Info] 数据处理完成: {}'.format(img_dir))


def parse_args():
    """
    处理脚本参数，支持相对路径
    :return: in_folder 输入文件夹, out_folder 输出文件夹, size 尺寸, n_prc 进程数
    """
    parser = argparse.ArgumentParser(description='压缩图片脚本')
    parser.add_argument('-i', dest='in_folder', required=True, help='输入文件夹', type=str)
    parser.add_argument('-p', dest='n_prc', required=False, default=40, help='进程数', type=str)
    parser.add_argument('-s', dest='size', required=False, default=50, help='最小边长', type=str)
    args = parser.parse_args()

    in_folder = args.in_folder
    size = int(args.size)
    n_prc = int(args.n_prc)
    print("文件路径：{}".format(in_folder))
    print("进程数: {}".format(n_prc))
    print("边长: {}".format(size))

    return in_folder, n_prc, size


def main():
    arg_in, n_prc, size = parse_args()
    check_error(arg_in, n_prc, size)


if __name__ == '__main__':
    main()