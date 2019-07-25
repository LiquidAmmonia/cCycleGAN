import torch
import torch.nn as nn

import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler

import numpy as np
import torch
import PIL.Image
import matplotlib.pyplot as plt


###############################################################################
# Functions
###############################################################################

def define_G():
    netG = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
        
    netG = GnetGenerator()


##############################################################################
# Classes
##############################################################################
