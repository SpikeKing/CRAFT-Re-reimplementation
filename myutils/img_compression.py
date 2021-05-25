#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/8
"""

import os
import cv2
import shutil
import argparse

from multiprocessing.pool import Pool


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


def compress_img(in_path, out_path, size=1024):
    # img = Image.open(in_path)
    # img.thumbnail((size, size))
    # img.save(out_path)
    try:
        img = cv2.imread(in_path)
        h, w, _ = img.shape
        if h >= w:
            w = int(w * size / h)
            h = size
        else:
            h = int(h * size / w)
            w = size
        img = cv2.resize(img, (w, h))
        cv2.imwrite(out_path, img)
        print('Processed: {}'.format(out_path))
    except Exception as e:
        print('[Error] error: {}, {}'.format(in_path, e))


def process_folder(in_folder, out_folder, size=1024, n_prc=20):
    mkdir_if_not_exist(out_folder)  # 创建文件夹
    path_list, name_list = traverse_dir_files(in_folder)
    print('[Info] 读取图像完成! {}'.format(len(path_list)))
    pool = Pool(processes=n_prc)  # 多线程下载

    for in_path, name in zip(path_list, name_list):
        out_path = os.path.join(out_folder, name)
        pool.apply_async(compress_img, (in_path, out_path, size))

    pool.close()
    pool.join()

    print('全部处理完成')


def parse_args():
    """
    处理脚本参数，支持相对路径
    :return: in_folder 输入文件夹, out_folder 输出文件夹, size 尺寸, n_prc 进程数
    """
    parser = argparse.ArgumentParser(description='压缩图片脚本')
    parser.add_argument('-i', dest='in_folder', required=True, help='输入文件夹', type=str)
    parser.add_argument('-o', dest='out_folder', required=True, help='输出文件夹', type=str)
    parser.add_argument('-s', dest='size', required=False, default=1024, help='最长边', type=str)
    parser.add_argument('-p', dest='n_prc', required=False, default=20, help='进程数', type=str)
    args = parser.parse_args()

    in_folder = args.in_folder
    print("文件路径：%s" % in_folder)

    out_folder = args.out_folder
    print("输出文件夹：%s" % out_folder)
    size = int(args.size)
    n_prc = int(args.n_prc)

    print('图片尺寸: {}, 进程数: {}'.format(size, n_prc))

    return in_folder, out_folder, size, n_prc


def main():
    arg_img, arg_out, size, n_prc = parse_args()
    mkdir_if_not_exist(arg_out)  # 新建文件夹
    process_folder(arg_img, arg_out, size, n_prc)


if __name__ == '__main__':
    main()
