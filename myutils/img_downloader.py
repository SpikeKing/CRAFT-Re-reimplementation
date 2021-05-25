#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/9
"""

import argparse
import os
import shutil
from datetime import datetime
from multiprocessing.pool import Pool

import requests


def get_current_time_str():
    """
    输入当天的日期格式, 20170718_1137
    :return: 20170718_1137
    """
    return datetime.now().strftime('%Y%m%d%H%M%S')


logfile = 'download_log_{}.log'.format(get_current_time_str())  # 日志文件


def download_img(img_url, out_folder, imgs_names, img_name=None):
    """
    下载图片
    :param img_url: 图片URL
    :param out_folder: 输出文件夹
    :param imgs_names: 已有图片
    :param img_name: 图片名称
    :return: None
    """
    if not img_name:
        img_name = img_url.split('/')[-1]  # 图片文件名

    if img_name in imgs_names:
        print_info('图片已存在: %s' % img_name)
        return

    img_data = requests.get(img_url).content

    out_file = os.path.join(out_folder, img_name)  # 输出文件

    with open(out_file, 'wb') as hl:
        hl.write(img_data)
        print_info('图片已下载: %s' % img_url)


def download_imgs_for_mp(img_file, out_folder, n_prc=10):
    """
    多线程下载
    :param img_file: 图片文件
    :param out_folder: 输出文件夹
    :param n_prc: 进程数, 默认40个
    :return: None
    """
    n_prc = int(n_prc)

    print_info('进程总数: %s' % n_prc)

    data_list = read_file(img_file)
    print_info('文件数: %s' % len(data_list))

    _, imgs_names = traverse_dir_files(out_folder)

    pool = Pool(processes=n_prc)  # 多线程下载
    for (index, data) in enumerate(data_list):
        items = data.split(',')
        if len(items) == 2:
            path, img_name = items
        else:
            path, img_name = data, None

        if img_name:
            pool.apply_async(download_img, (path, out_folder, imgs_names, img_name))
        else:
            pool.apply_async(download_img, (path, out_folder, imgs_names))

    pool.close()
    pool.join()

    # _, imgs_names = traverse_dir_files(out_folder)
    # print_info('图片总数: %s' % len(imgs_names))
    print_info('全部下载完成')


def parse_args():
    """
    处理脚本参数，支持相对路径
    img_file 文件路径，默认文件夹：img_downloader/urls
    out_folder 输出文件夹，默认文件夹：img_data
    :return: arg_img，文件路径；out_folder，输出文件夹
    """
    parser = argparse.ArgumentParser(description='下载数据脚本')
    parser.add_argument('-i', dest='img_file', required=True, help='文件路径', type=str)
    parser.add_argument('-o', dest='out_folder', required=True, help='输出文件夹', type=str)
    parser.add_argument('-p', dest='n_prc', required=False, default=20, help='进程数', type=str)

    args = parser.parse_args()

    arg_img = args.img_file
    print_info("文件路径：%s" % arg_img)

    arg_out = args.out_folder
    print_info("输出文件夹：%s" % arg_out)

    arg_npc = args.n_prc
    print_info("进程数：%s" % arg_npc)
    return arg_img, arg_out, arg_npc


def write_line(file_name, line):
    """
    将行数据写入文件
    :param file_name: 文件名
    :param line: 行数据
    :return: None
    """
    if file_name == "":
        return
    with open(file_name, "a+", encoding='utf8') as fs:
        if type(line) is (tuple or list):
            fs.write("%s\n" % ", ".join(line))
        else:
            fs.write("%s\n" % line)


def print_info(log_str):
    """
    打印日志
    :param log_str: 日志信息
    :return: None
    """
    log_str = u'[Info {}] {}'.format(get_current_time_str(), str(log_str))
    write_line(logfile, log_str)
    print(log_str)


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


def sort_two_list(list1, list2):
    """
    排序两个列表
    :param list1: 列表1
    :param list2: 列表2
    :return: 排序后的两个列表
    """
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def read_file(data_file, mode='more'):
    """
    读文件, 原文件和数据文件
    :return: 单行或数组
    """
    try:
        with open(data_file, 'r') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                output = [o.strip() for o in output]
                return output
            else:
                return list()
    except IOError:
        return list()


def main():
    """
    入口函数
    """
    arg_img, arg_out, arg_npc = parse_args()
    mkdir_if_not_exist(arg_out)  # 新建文件夹
    download_imgs_for_mp(arg_img, arg_out, arg_npc)


if __name__ == '__main__':
    main()
