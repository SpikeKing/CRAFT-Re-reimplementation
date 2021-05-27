#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 25.5.21
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'mydata')

SYNTH_TEXT_PATH = os.path.join(ROOT_DIR, '..', 'datasets', 'SynthText')
ICDAR_2015_PATH = os.path.join(ROOT_DIR, '..', 'datasets', 'craft', 'ICDAR_2015')
PRETRAIN_PATH = os.path.join(DATA_DIR, 'models', 'vgg16_bn-6c64b313.pth')
