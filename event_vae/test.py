import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import loss

from torch.utils.tensorboard import SummaryWriter

from event_ae import EventAE
from chamfer import ChamferDistance, ChamferLoss
from dataloader import EventStreamDataset, EventStreamArray
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    type=str,
    default="data/MSTrain_bytestream.txt",
    help="training data filename",
)
parser.add_argument("--batch_size", type=int,
                    default=1000, help="input batch size")
parser.add_argument("--batch_num", type=int, default=50,
                    help="number of batches")
parser.add_argument("--data_len", type=int, default=2,
                    help="event element length")
parser.add_argument("--tcode", type=bool, default=False,
                    help="consider timestamps")
parser.add_argument(
    "--nepoch", type=int, default=5000, help="number of epochs to train for"
)
parser.add_argument("--latent_size", type=int, default=8)
parser.add_argument(
    "--rec_loss",
    type=str,
    default="huber",
    help="type of loss: mse, huber, bce, chamfer",
)
parser.add_argument(
    "--decoder", type=str, default="image", help="decoder type: stream or image"
)
parser.add_argument("--outf", type=str, default="weights",
                    help="output folder")
parser.add_argument("--model", type=str, default="", help="model path")
parser.add_argument(
    "--norm_type",
    type=str,
    default="none",
    help="normalization type: scale: [0, 1]; center: [-1, 1]",
)
parser.add_argument("--arch", type=str, default="vanilla")

opt = parser.parse_args()
print(opt)


def blue(x): return "\033[94m" + x + "\033[0m"


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

writer = SummaryWriter("runs/str_to_img_test")

# Params:
# n_events, data_size for stream decoder
# Height, width for image decoder

H = 32
W = 32
params = [H, W]

event_dataset = EventStreamArray(
    opt.input_file, opt.batch_num, opt.batch_size, opt.data_len
)

"""
batch_size_total = opt.batch_size * opt.batch_num
train_loader = data.DataLoader(
    event_dataset,
    batch_size=batch_size_total,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)
"""

data_utils = EventDataUtils(32, 32, opt.norm_type)

enet = EventAE(
    opt.data_len, opt.latent_size, params, decoder=opt.decoder, norm_type=opt.norm_type
)
if opt.model != "":
    enet.load_state_dict(torch.load(opt.model))
enet.cuda()

init = True
event_np_stack = np.empty([opt.batch_num, opt.batch_size, 4], dtype=np.float32)
frames_stack = np.empty([opt.batch_num, H * W], dtype=np.float32)

if opt.data_len == 3:
    pol = True
else:
    pol = False


with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        # events = data_utils.normalize(EventExtractor(data, batch_num=1))

        idx = random.randint(0, 1000000)
        events = data_utils.normalize(event_array.get_event_stack(idx))
        events = Variable(events)
        events = events.transpose(2, 1)
        events = events.cuda()

        recon, z = enet(events)

        events = events.transpose(2, 1).contiguous()

        if opt.decoder == "stream":
            recon = recon.transpose(2, 1).contiguous()

        data_utils.compare_frames(events, recon)
