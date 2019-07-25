import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models, transforms
import sys
import argparse
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import PIL.Image
from dataloader.dataloader import CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='',
                    help='path to images (should have subfolders trainA, trainB, valA, valB, testA, testB)')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imgSize', type=int, default=128, help='scale img to this size')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
#parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
#parser.add_argument('--train_display_id', type=int, default=1, help='window id of the web display')
#parser.add_argument('--val_display_id', type=int, default=10, help='window id of the web display')
#parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
#parser.add_argument('--display_single_pane_ncols', type=int, default=4,
#                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
#parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
#parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
#                         help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
#parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
#                         help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width]')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
#parser.add_argument('--no_flip', action='store_true', help='use dropout for the generator')
parser.add_argument('--resume', default = '')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
#parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
#parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
#parser.add_argument('--niter_decay', type=int, default=300,
#                         help='# of iter to linearly decay learning rate to zero')
#parser.add_argument('--print_iter', type=int, default=50, help='frequency of showing training results on console')
#parser.add_argument('--display_iter', type=int, default=50, help='frequency of showing training results on console')
#parser.add_argument('--save_iter', type=int, default=100, help='frequency of showing training results on console')
#parser.add_argument('--ckpt_path', default='')

opt = parser.parse_args()
print(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

opt.phase = 'val'
val_data_loader = CreateDataLoader(opt)
val_dataset = val_data_loader.load_data
val_dataset_size = len(val_data_loader)

## define models 
# X is the source and sketch domain
# Y is the target and photo domain
# An auxiliary discriminator DA to detect the feature A 
GX =