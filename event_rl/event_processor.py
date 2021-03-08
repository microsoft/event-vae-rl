import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from copy import deepcopy
import random
import time

from event_vae import EventVAE


class EventProcessor:
    def __init__(
        self,
        data_len,
        image_size,
        pretrained_weights,
        ls,
        tc,
        debug,
        noise_level=0.0,
        sparsity=0.0,
    ):
        self.input_size = data_len

        self.dt = 1000
        # Set image resolution
        self.H = image_size[0]
        self.W = image_size[1]

        self.frame = None
        self.recon = None
        self.z = None
        self.z_prev = None

        # Initialize event VAE and load weights
        self.en = EventVAE(
            data_len, ls, [self.H, self.W], decoder="image", norm_type="none", tc=tc
        )

        checkpoint = torch.load(pretrained_weights)
        self.en.load_state_dict(checkpoint)
        self.en.cuda()
        self.en.eval()

        self.debug = debug
        if self.debug:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        # Two kinds of noise simulated: probability of events either not firing when they
        # should (off_pix_prob), or firing when they should not (noise_pix_prob)
        self.off_pix_prob = sparsity
        self.noise_pix_prob = noise_level

        self.idx = 0

    def convertStream(self, events, dt, n_pix_ev):
        """
        Convert an input ordered stream into a latent representation
        """

        n_events_total = len(events)

        start_idx = 0
        t_start = events[start_idx][0]
        t_curr = t_start
        frame_events = []
        idx = start_idx

        # Create empty frame
        self.frame = np.zeros((self.H, self.W, 1), np.float32)
        n_events = n_events_total

        t_final = events[n_events - 1][0]

        dt = t_final - t_start

        if dt < 1e-3:
            dt = 1e-3

        # Create empty frame
        self.frame = np.zeros((self.H, self.W, 1), np.float32)
        n_events = n_events_total

        # Precompute indices at which events should not fire - for off pixel noise
        off_pixels = np.random.choice(
            a=[1, 0],
            p=[self.off_pix_prob, 1 - self.off_pix_prob],
            size=(self.H * self.W,),
        )

        # Precompute indices at which events should fire arbitrarily - for on pixel noise
        noise_pixels = np.random.choice(
            a=[1, 0],
            p=[
                self.noise_pix_prob * (n_pix_ev / 4096),
                1 - self.noise_pix_prob * (n_pix_ev / 4096),
            ],
            size=((self.H * self.W),),
        )
        noise_pixels_idx = np.nonzero(noise_pixels)[0]

        # Stack events
        while idx - start_idx < n_events:
            e_curr = deepcopy([*events[idx]])

            # Skip locations where we wish to simulate off pixel noise
            if off_pixels[e_curr[1] * e_curr[2]]:
                idx += 1
                continue

            frame_events.append(e_curr)

            # Timestamps are relative to window of time observed in the event data
            t_relative = float(t_final - e_curr[0]) / dt
            frame_events[-1][0] = t_relative
            t_curr = float(e_curr[0])
            idx += 1

        for i in range(len(noise_pixels_idx)):
            # Inject random noise pixel firings if required

            ev_noise = [
                np.random.uniform(0.0, 1.0),
                int(noise_pixels_idx[i] // 64),
                int(noise_pixels_idx[i] % 64),
                random.choice([-1, 1]),
            ]
            frame_events.append(ev_noise)

        idx = 0
        for e in frame_events:  # T, X, Y, P
            self.frame[int(e[1]), int(e[2])] += (
                e[3] * e[0] if self.input_size == 3 else 1
            )
            idx += 1

        self.frame = np.clip(self.frame, -1, 1)
        events_np = np.asarray(frame_events, dtype=np.float32)

        if events_np.shape[0] == 0:
            return torch.zeros([1, 8])

        events_no_t = events_np[:, 1 : self.input_size + 1]
        timestamps = torch.from_numpy(events_np[:, 0].reshape(1, events_no_t.shape[0]))

        # Pass data through autoencoder
        x = torch.from_numpy(events_no_t).reshape(
            1, events_no_t.shape[0], self.input_size
        )
        x = x.transpose(2, 1).cuda()
        timestamps = timestamps.cuda()

        # Draw reconstructions if needed
        if self.debug:
            x_o, self.z, _, _ = self.en.forward(x, timestamps)
            self.recon = x_o.reshape(self.H, self.W, 1).cpu().detach().numpy()
            self.plotChange()
        else:
            self.z = self.en.encode(x, timestamps)

        # RL algorithm operates on the latent vector
        return self.z

    def plotChange(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.imshow(self.frame[:, :, 0], cmap="viridis")
        self.ax2.imshow(self.recon[:, :, 0], cmap="viridis")
        # plt.savefig(f"frame_{self.idx}.png")
        # self.idx += 1
        plt.pause(0.001)


if __name__ == "__main__":
    pass
