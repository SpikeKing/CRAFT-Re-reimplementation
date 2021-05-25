#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 2020/8/15
"""


def draw_plt_histograms(data_list, label_list, title=""):
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    # Fake dataset
    height = data_list
    bars = label_list
    y_pos = np.arange(len(bars))

    fig = plt.gcf()
    fig.set_size_inches(20, 8)

    plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))

    # Add title and axis names
    plt.title(title)
    # plt.xlabel('categories')
    # plt.ylabel('values')

    # Limits for the Y axis
    max_y = (max(data_list) / 500 + 1) * 500
    plt.ylim(0, max_y)

    # Create names
    plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()


def main():
    draw_plt_histograms(None, None)


if __name__ == '__main__':
    main()
