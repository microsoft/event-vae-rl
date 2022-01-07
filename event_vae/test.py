import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

from event_vae import EventVAE
from dataloader import EventStreamArray
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    type=str,
    default="data/gates.txt",
    help="training data filename",
)
parser.add_argument("--batch_size", type=int,
                    default=1000, help="input batch size")
parser.add_argument("--batch_num", type=int, default=1,
                    help="number of batches")
parser.add_argument("--data_len", type=int, default=3,
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
parser.add_argument("--output_dir", type=str, default="weights",
                    help="output folder")
parser.add_argument("--model", type=str, default="weights/evae_racing.pt", help="model path")
parser.add_argument(
    "--norm_type",
    type=str,
    default="none",
    help="normalization type: scale: [0, 1]; center: [-1, 1]",
)

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Params:
# n_events, data_size for stream decoder
# Height, width for image decoder

H = 64
W = 64
params = [H, W]

event_dataset = EventStreamArray(
    opt.input_file, opt.batch_num, opt.batch_size, opt.data_len
)

data_utils = EventDataUtils(H, W, opt.norm_type)

enet = EventVAE(
    opt.data_len, opt.latent_size, params, decoder=opt.decoder)

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

step_size = 50
start_idx = 0

event_dataset = EventStreamArray(opt.input_file, opt.batch_num, step_size, opt.data_len)

with torch.no_grad():
    idx = random.randint(0, len(event_dataset.events) - 200000)

    while step_size < 4000:
        event_data = event_dataset.get_event_batch_idx(idx, step_size)
        gt = data_utils.create_frame(event_data, pol, opt.tcode)
        gt = torch.from_numpy(gt).cuda()
        events, timestamps = event_dataset.extract(event_data.reshape(1, step_size, 4))
        events = events.transpose(2, 1)
        events = events.cuda()

        recon, z, mu, logvar = enet(events)

        data_utils.ax1.cla()
        data_utils.ax2.cla()

        z = z.cpu().numpy()

        fig = data_utils.compare_frames(gt, recon)
        # data_utils.ax3.scatter(z[0][0], z[0][1], z[0][2], 20)
        # data_utils.ax3.set_aspect("equal", adjustable="box")
        plt.draw()
        plt.pause(0.001)
        step_size += 50
