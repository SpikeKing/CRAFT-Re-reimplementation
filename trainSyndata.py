import os
import argparse
import random
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from craft import CRAFT
from data_loader import Synth80k, HWFakeDB
from eval.script import getresult
###import file#######
from mseloss import Maploss
from test import test

from myutils.project_utils import mkdir_if_not_exist
from root_dir import SYNTH_TEXT_PATH, DATA_DIR, ROOT_DIR

# 3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size of training')
# parser.add_argument('--cdua', default=True, type=str2bool,
# help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # gaussian = gaussion_transform()
    # box = scio.loadmat('/data/CRAFT-pytorch/syntext/SynthText/gt.mat')
    # bbox = box['wordBB'][0][0][0]
    # charbox = box['charBB'][0]
    # imgname = box['imnames'][0]
    # imgtxt = box['txt'][0]

    # dataloader = syndata(imgname, charbox, imgtxt)
    # dataloader = Synth80k('/data/CRAFT-pytorch/syntext/SynthText/SynthText', target_size = 768)

    # ???????????????
    # dataloader = Synth80k(SYNTH_TEXT_PATH, target_size=768)  # ????????????

    # ????????????????????????
    HW_TEXT_PATH = os.path.join(ROOT_DIR, '..', 'datasets', 'hw_generator_20210608')
    dataloader = HWFakeDB(HW_TEXT_PATH, target_size=768)

    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=16,
        # batch_size=1,  # ??????
        shuffle=True,
        # shuffle=False,  # ??????
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    # batch_syn = iter(train_loader)
    # prefetcher = data_prefetcher(dataloader)
    # input, target1, target2 = prefetcher.next()
    # print(input.size())
    net = CRAFT()
    # net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/CRAFT_net_050000.pth')))
    # net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/1-7.pth')))
    pretrain_model_path = os.path.join(DATA_DIR, 'models', 'craft_mlt_25k.pth')
    net.load_state_dict(copyStateDict(torch.load(pretrain_model_path)))  # ?????????????????????
    # net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
    # realdata = realdata(net)
    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size = 768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)

    # ----- CUDA?????? -----
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3]).cuda()
    cudnn.benchmark = True
    # ----- CUDA?????? -----

    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size=768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    # criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    net.train()

    step_index = 0

    loss_time = 0
    loss_value = 0
    compare_loss = 1

    out_dir = os.path.join(DATA_DIR, 'synweights')
    mkdir_if_not_exist(out_dir)

    for epoch in range(1000):
        loss_value = 0
        # if epoch % 50 == 0 and epoch != 0:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        for index, (images, gh_label, gah_label, mask, _) in enumerate(train_loader):
            if index % 20000 == 0 and index != 0:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            # real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)

            # syn_images, syn_gh_label, syn_gah_label, syn_mask = next(batch_syn)
            # images = torch.cat((syn_images,real_images), 0)
            # gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            # gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            # mask = torch.cat((syn_mask, real_mask), 0)

            # affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)

            images = Variable(images.type(torch.FloatTensor)).cuda()
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()
            # affinity_mask = affinity_mask.type(torch.FloatTensor)
            # affinity_mask = Variable(affinity_mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'
                      .format(epoch, index, len(train_loader), et - st, loss_value / 2))
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth'

            if index % 5000 == 0 and index != 0:
                print('Saving state, index:', index)
                # torch.save(net.module.state_dict(), '/data/CRAFT-pytorch/synweights/synweights_' + repr(index) + '.pth')
                # test('/data/CRAFT-pytorch/synweights/synweights_' + repr(index) + '.pth')
                synweights_path = os.path.join(out_dir, 'synweights_' + repr(index) + '.pth')
                torch.save(net.module.state_dict(), synweights_path)
                test(synweights_path)
                # test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                getresult()
