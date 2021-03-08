import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import loss

from torch.utils.tensorboard import SummaryWriter

from event_vae import EventVAE
from dataloader import EventStreamArray
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    type=str,
    default="data/esim_RacingCourse.txt",
    help="training data filename",
)
parser.add_argument("--batch_size", type=int, default=2000, help="input batch size")
parser.add_argument("--batch_num", type=int, default=50, help="number of batches")
parser.add_argument("--data_len", type=int, default=3, help="event element length")
parser.add_argument("--tcode", action="store_true", help="consider timestamps")
parser.add_argument(
    "--nepoch", type=int, default=20000, help="number of epochs to train for"
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
parser.add_argument("--outf", type=str, default="weights", help="output folder")
parser.add_argument("--model", type=str, default="", help="model path")
parser.add_argument(
    "--norm_type",
    type=str,
    default="none",
    help="normalization type: scale: [0, 1]; center: [-1, 1]",
)
parser.add_argument("--lamda", type=float, default=1e-5)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--scaling", type=str, default="anneal")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--output_dir", type=str, default="./weights")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

writer = SummaryWriter(opt.output_dir)
opt.tcode = True

# Params:
# n_events, data_size for stream decoder
# Height, width for image decoder

H = 64
W = 64
params = [H, W]

# Create a custom dataset from the input event data file

event_dataset = EventStreamArray(
    opt.input_file, opt.batch_num, opt.batch_size, opt.data_len
)

# EventDataUtils can be used to normalize the data if needed
data_utils = EventDataUtils(H, W, opt.norm_type)

# Initialize VAE model
enet = EventVAE(opt.data_len, opt.latent_size, params, decoder=opt.decoder)
if opt.model != "":
    enet.load_state_dict(torch.load(opt.model))
enet.cuda()

optimizer = optim.Adam(enet.parameters(), lr=0.001, weight_decay=opt.wd)
loss = loss.LossModule(opt.rec_loss, ae_type="variational", lamda=opt.lamda)

init = True

# Init some variables for batches of events and corresponding frames
event_np_stack = np.empty([opt.batch_num, opt.batch_size, 4], dtype=np.float32)
frames_stack = np.empty([opt.batch_num, H * W], dtype=np.float32)

if opt.data_len == 3:
    pol = True
else:
    pol = False

for epoch in range(opt.nepoch):
    optimizer.zero_grad()

    # Obtain a 'stack' of events each starting from a random index,
    # compute corresponding frames for reconstruction

    event_data = event_dataset.get_event_stack(event_np_stack)
    data_utils.create_frame_stack(event_data, frames_stack, pol, opt.tcode)
    gt = torch.from_numpy(frames_stack).cuda()

    # Split event data into spatial and temporal parts
    events, timestamps = event_dataset.extract(event_data)
    timestamps = timestamps.cuda()

    events = events.transpose(2, 1)
    events = events.cuda()
    recon, z, mu, logvar = enet(events, timestamps)

    # Compute reconstruction loss
    train_loss = loss.compute_loss(
        epoch, events, gt, recon, mu, logvar, scaling=opt.scaling
    )

    # Train
    train_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # Log to tensorboard
        print("Writing")
        writer.add_scalar("training loss", train_loss.item(), epoch)
        writer.add_scalar("loss 1", loss.loss1.item(), epoch)
        writer.add_scalar("loss 2", loss.loss2.item(), epoch)
        writer.add_scalar("beta", loss.scale_factor, epoch)

        sample_gt = gt[0]
        writer.add_figure(
            "reconstructed vs actual",
            data_utils.compare_frames(sample_gt, recon[0]),
            global_step=epoch,
        )
        torch.save(
            enet.state_dict(), os.path.join(opt.output_dir, "evae_%d.pt" % (epoch)),
        )
