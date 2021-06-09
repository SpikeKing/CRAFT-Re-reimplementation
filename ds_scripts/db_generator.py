#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 2.6.21
"""

import os
import sys

from multiprocessing.pool import Pool
from PIL import ImageDraw, ImageFont, Image

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.cv4png_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DbGenerator(object):
    """
    数据集生成类
    """

    def __init__(self, is_test=False):
        self.words_dict = DbGenerator.prepare_words_dict_imgs_mp(is_test)
        self.news_lines = DbGenerator.prepare_news_lines()
        self.white_bkg_imgs = DbGenerator.prepare_bkg_imgs(is_white=True)
        self.black_bkg_imgs = DbGenerator.prepare_bkg_imgs(is_white=False)
        self.out_dir = os.path.join(
            ROOT_DIR, '..', 'datasets', 'hw_generator_{}'.format(get_current_day_str()))
        mkdir_if_not_exist(self.out_dir)
        self.is_test = is_test

    @staticmethod
    def clean_news_file():
        """
        清洗头条的新闻文本
        """
        file_path = os.path.join(DATA_DIR, 'toutiao_cat_data.txt')
        print('[Info] 输入文件: {}'.format(file_path))
        out_path = os.path.join(DATA_DIR, 'toutiao_cat_data.clean.txt')
        print('[Info] 输出文件: {}'.format(out_path))
        data_lines = read_file(file_path)

        news_list = []

        # 样本: 6552368441838272771_!_101_!_news_culture_!_发酵床的垫料种类有哪些？哪种更好？_!_
        for data_line in data_lines:
            items = data_line.split("_!_")
            news_str = items[3]
            news_list.append(news_str)

        print('[Info] 样本数: {}'.format(len(news_list)))
        write_list_to_file(out_path, news_list)

    @staticmethod
    def prepare_news_lines(is_random=False):
        """
        创建新闻的文本行
        """
        news_file = os.path.join(DATA_DIR, 'toutiao_cat_data.clean.txt')
        news_lines = read_file(news_file)
        if not is_random:
            random.seed(47)
        random.shuffle(news_lines)
        print('[Info] 初始化新闻文本!')
        return news_lines

    @staticmethod
    def prepare_words_dict_urls():
        """
        创建文字字典
        """
        words_dir = os.path.join(DATA_DIR, 'chinese_word_files')
        paths_list, names_list = traverse_dir_files(words_dir)

        words_dict = dict()
        for path, name in zip(paths_list, names_list):
            word = name.split('.')[0]
            urls = read_file(path)
            words_dict[word] = urls

        print('[Info] 初始化字典完成!')
        return words_dict

    @staticmethod
    def prepare_words_dict_imgs(is_test):
        """
        创建文字字典图像
        """
        print('[Info] 初始化图像字典开始!')
        s_time = time.time()
        words_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'chinese_words_1000')
        paths_list, names_list = traverse_dir_files(words_dir)
        print('[Info] \t字数: {}'.format(len(paths_list)))
        words_dict = collections.defaultdict(list)
        paths_list, names_list = shuffle_two_list(paths_list, names_list)

        count = 0
        for path, name in zip(paths_list, names_list):
            word = path.split('/')[-2]  # 字
            img_bgr = cv2.imread(path)
            words_dict[word].append(img_bgr)
            count += 1
            if count % 10000 == 0:
                print('[Info] \tcount: {}'.format(count))
                if is_test:
                    break  # 测试
        print('[Info] \t初始化图像字典完成! {}'.format(time.time() - s_time))

        return words_dict

    @staticmethod
    def cv2_imread_worker(param):
        idx, path = param
        img_bgr = cv2.imread(path)
        if idx % 1000 == 0:
            print('[Info] \tcount: {}'.format(idx))
        return img_bgr

    @staticmethod
    def prepare_words_dict_imgs_mp(is_test):
        """
        创建文字字典图像
        """
        print('[Info] 初始化图像字典开始!')
        s_time = time.time()
        words_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'chinese_words_1000')
        paths_list, names_list = traverse_dir_files(words_dir)
        print('[Info] \t字数: {}'.format(len(paths_list)))
        words_dict = collections.defaultdict(list)
        paths_list, names_list = shuffle_two_list(paths_list, names_list)

        test_num = 10000
        if is_test:
            paths_list, names_list = paths_list[:test_num], names_list[:test_num]

        count = 0
        pool = Pool(processes=100)
        word_list = []
        param_list = []
        for idx, path in enumerate(paths_list):
            param_list.append((idx, path))
        buffer_list = pool.map(DbGenerator.cv2_imread_worker, param_list)
        for path, name in zip(paths_list, names_list):
            word = path.split('/')[-2]  # 字
            buffer_list.append(buffer_list)
            word_list.append(word)
            if is_test and count == test_num:
                break  # 测试

        for buffer, word in zip(buffer_list, word_list):
            img_bgr = buffer
            words_dict[word].append(img_bgr)

        print('[Info] \t初始化图像字典完成! {}'.format(time.time() - s_time))
        return words_dict

    @staticmethod
    def prepare_bkg_imgs(is_white=True):
        if is_white:
            bkg_dir = os.path.join(DATA_DIR, "bkgs", "white_board")
        else:
            bkg_dir = os.path.join(DATA_DIR, "bkgs", "black_board")

        bkg_paths, _ = traverse_dir_files(bkg_dir)

        bkg_imgs = []
        for bkg_path in bkg_paths:
            bkg_img = cv2.imread(bkg_path)
            bkg_img = resize_img_fixed(bkg_img, 1500, is_height=False)  # 按框标注
            bkg_imgs.append(bkg_img)

        print('[Info] {}色 背景总数: {}'.format("白" if is_white else "黑", len(bkg_imgs)))
        return bkg_imgs

    @staticmethod
    def get_word_png_url(words_dict, word, idx=-1, is_white=True):
        """
        输入汉字，输出汉字对应的PNG图像
        """
        urls = words_dict[word]
        n_word = len(urls)
        if 0 <= idx < n_word:
            url = urls[idx]
        else:
            idx = random.randint(0, n_word - 1)
            url = urls[idx]
        _, img_bgr = download_url_img(url)  # 下载图像

        img_new = resize_with_padding(img_bgr, 144)
        # show_img_bgr(img_new)

        img_bold = improve_img_bold(img_new, times=10)
        img_word = img_white_2_png(img_bold, is_white=is_white)
        return img_word

    @staticmethod
    def get_word_png_img(words_dict, word, idx=-1, is_white=True):
        """
        输入汉字，输出汉字对应的PNG图像
        """
        imgs = words_dict[word]
        n_word = len(imgs)
        if 0 <= idx < n_word:
            img_bgr = imgs[idx]
        else:
            idx = random.randint(0, n_word - 1)
            img_bgr = imgs[idx]

        img_new = resize_with_padding(img_bgr, 144)
        # show_img_bgr(img_new)

        img_bold = improve_img_bold(img_new, times=10)
        img_word = img_white_2_png(img_bold, is_white=is_white)
        return img_word

    @staticmethod
    def get_symbol_word(text):
        """
        绘制中文印刷体或其他，使用Pillow代替OpenCV，因为Pillow对于中文更加友好
        下载字体文件simsun.ttc
        """
        bkg_shape = (144, 144, 3)  # 图像的背景尺寸

        # 字体和大小
        text_size = 128  # 图像的文字尺寸
        font_path = os.path.join(DATA_DIR, "fonts", "simsun.ttc")  # <== 这里是宋体路径
        font = ImageFont.truetype(font_path, text_size)

        # 背景
        img = np.ones(bkg_shape, np.uint8) * 255
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        # 计算图像的中心位置
        w, h = draw.textsize(text, font=font)
        text_x = int((img.shape[1] - w) / 2)
        text_y = int((img.shape[0] - h) / 2)

        # 绘制图像，转换为OpenCV的通道
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # show_img_bgr(img)
        return img

    @staticmethod
    def get_news_line(words_dict, news_str, is_white=True):
        """
        将新闻转换为手写的PNG图像
        """
        # print('[Info] news_line: {}'.format(news_line))
        png_list, char_list = [], []
        for c_char in news_str:
            if c_char not in words_dict:
                continue
                # 生成印刷体代替
                # img_bgr = self.get_symbol_word(c_char)
                # img_png = img_white_2_png(img_bgr)
            else:
                img_png = DbGenerator.get_word_png_img(words_dict, c_char, is_white=is_white)
                char_list.append(c_char)
            # show_img_png(img_png)
            png_list.append(img_png)
        img_large_png = merge_imgs(png_list, cols=len(png_list), rows=1)
        # show_img_png(img_large_png, save_name="large.png")
        return img_large_png, char_list

    @staticmethod
    def check_width_thres(pasted_img, bkg_img, s_x, s_y):
        ph, pw, _ = pasted_img.shape
        bh, bw, _ = bkg_img.shape
        if ph + s_y > bh or pw + 2 * s_x > bw:
            return False
        else:
            return True

    @staticmethod
    def generate_news_image_worker(param):
        idx, out_dir, words_dict, news_lines, white_bkg_imgs, black_bkg_imgs = param
        # print('[Info] 开始 idx: {}'.format(idx))
        try:
            DbGenerator.generate_news_image(
                idx, out_dir, words_dict, news_lines, white_bkg_imgs, black_bkg_imgs)
            print('[Info] 完成 idx: {}'.format(idx))
        except Exception as e:
            print('[Exception] e: {}'.format(e))
            print('[Exception] 完成 idx: {}'.format(idx))

    @staticmethod
    def generate_news_image(idx, out_dir, words_dict, news_lines, white_bkg_imgs, black_bkg_imgs):
        """
        生成新闻图像
        """
        idx = str(idx).zfill(7)
        print("[Info] idx: {}".format(idx) + '-' * 50)
        if random_prob(0.5):
            bkg_imgs = white_bkg_imgs
            is_white = True
        else:
            bkg_imgs = black_bkg_imgs
            is_white = False

        # 随机背景
        bkg_idx = random.randint(0, len(bkg_imgs))
        bkg_idx = bkg_idx % len(bkg_imgs)
        bkg_img = bkg_imgs[bkg_idx]
        # print('[Info] num of bkgs: {}'.format(len(bkg_imgs)))
        # print('[Info] bkg_img: {}'.format(bkg_img.shape))

        # 随机文本行数
        random.shuffle(news_lines)
        num_news = random.randint(2, 10)  # 随机新闻数量

        # 固定尺寸参数
        word_size = 40
        start_x, start_y = 200, 200

        words_bboxes = []
        line_idx = 0
        for news_idx in range(len(news_lines)):
            if news_idx == num_news:
                break
            news_str = news_lines[news_idx]
            if len(news_str) > 30:  # 最多30个字
                news_str = news_str[:30]
            line_png, char_list = DbGenerator.get_news_line(words_dict, news_str, is_white=is_white)
            line_png = resize_img_fixed(line_png, word_size, is_height=True)
            # print('[Info] idx: {} line_png: {}'.format(news_idx, line_png.shape))
            start_y_tmp = start_y + news_idx * word_size  # 高度一直增加
            if not DbGenerator.check_width_thres(line_png, bkg_img, start_x, start_y_tmp):
                print('[Info] 超出界限: {}'.format(news_idx))
                continue
            bkg_img = paste_png_on_bkg(line_png, bkg_img, (start_x, start_y_tmp))
            line_idx += 1
            # 添加词
            for word_idx in range(len(char_list)):
                out_list = [char_list[word_idx], line_idx,
                            [start_x + word_idx * word_size,
                             start_y_tmp,
                             start_x + (word_idx + 1) * word_size,
                             start_y_tmp + word_size]]
                words_bboxes.append(json.dumps(out_list))

        out_name = "hwg_{}".format(idx)
        img_path = os.path.join(out_dir, out_name + ".jpg")
        lbl_path = os.path.join(out_dir, out_name + ".txt")
        # show_img_bgr(bkg_img, save_name=img_path)
        cv2.imwrite(img_path, bkg_img)
        create_file(lbl_path)
        write_list_to_file(lbl_path, words_bboxes)

    def generate_datasets(self):
        """
        生成数据集
        """
        num_of_sample = 500000
        random.seed(47)
        pool = Pool(processes=10)

        # 已处理的图像
        _, processed_names = traverse_dir_files(self.out_dir, ext="jpg")
        processed_nums = []
        for name in processed_names:
            processed_nums.append(int(name.split(".")[0].split('_')[-1]))

        params_list = []
        for idx in range(num_of_sample):
            if idx in processed_nums:
                print('[Info] 已处理: {}'.format(idx))
                continue
            params_list.append((idx, self.out_dir, self.words_dict, self.news_lines,
                                self.white_bkg_imgs, self.black_bkg_imgs))
            if self.is_test and idx == 10:
                break

        print('[Info] 样本数: {}'.format(len(params_list)))
        # pool.map(DbGenerator.generate_news_image_worker, params_list)
        for params in params_list:
            DbGenerator.generate_news_image_worker(params)
        print('[Info] 样本生成完成! num: {} path: {}'.format(len(params_list), self.out_dir))

    def check_data(self):
        lbls_paths, _ = traverse_dir_files(self.out_dir, ext="txt")
        imgs_paths, _ = traverse_dir_files(self.out_dir, ext="jpg")

        random.seed(47)
        idx = random.randint(0, len(lbls_paths) - 1)
        lbl_path, img_path = lbls_paths[idx], imgs_paths[idx]
        data_lines = read_file(lbl_path)
        line_dict = collections.defaultdict(list)

        img_bgr = cv2.imread(img_path)

        for data_line in data_lines:
            items = json.loads(data_line)
            line_num = items[1]
            word = items[0]
            bbox = items[2]
            line_dict[line_num].append([word, bbox])  # box转换成rec

        sample_words, sample_boxes = [], []
        for line_num in line_dict.keys():
            info_list = line_dict[line_num]
            words, bbox_list = "", []
            for info_data in info_list:
                word, bbox = info_data
                words += word
                bbox_list.append(bbox)
            img_bgr = draw_box_list(img_bgr, bbox_list, is_show=True, is_new=False)
            sample_words.append(words)
            print('[Info] words: {}'.format(words))

        cv2.imwrite("tmp.jpg", img_bgr)


def main():
    dg = DbGenerator(is_test=False)
    # dg.get_word_png("春", idx=5)
    # dg.get_word_img("美")
    dg.generate_datasets()
    # dg.check_data()


if __name__ == '__main__':
    main()
