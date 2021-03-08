import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class EventDataUtils:
    def __init__(self, H, W, norm_type):
        self.W = W - 1
        self.H = H - 1

        self.norm_type = norm_type
        self.decoder = "image"

        self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)

    def normalize(self, data):
        """
        Normalize the event data across pixel space if needed to help with VAE
        """
        if self.norm_type == "center":
            data[:, :, 0] = 2 * (data[:, :, 0]) / (self.H) - 1
            data[:, :, 1] = 2 * (data[:, :, 1]) / (self.W) - 1

        elif self.norm_type == "scale":
            data[:, :, 0] = data[:, :, 0] / (self.H)
            data[:, :, 1] = data[:, :, 1] / (self.W)

        return data

    def create_frame(self, data, pol=False, tc=False):
        """
        Create a frame from event data for reconstruction loss
        """
        frame = np.zeros((self.H + 1, self.W + 1, 1), dtype=np.float32)

        for i in range(data.shape[0]):
            if self.norm_type == "center":
                x_loc = int((data[i, 1] + 1) * self.H / 2)
                y_loc = int((data[i, 2] + 1) * self.W / 2)
            elif self.norm_type == "scale":
                x_loc = int((data[i, 1]) * self.H)
                y_loc = int((data[i, 2]) * self.W)
            else:
                x_loc = int(data[i, 1])
                y_loc = int(data[i, 2])

            if not pol and not tc:
                frame[x_loc, y_loc] = 1
                continue

            pix_val = 1
            if pol:
                # Image contains (-1, 1) values if polarity is considered
                pix_val = data[i, 3]

            if tc:
                # Scale polarity according to timestamp for a simple
                # representation to decode to

                pix_val = pix_val * data[i, 0]

            frame[x_loc, y_loc] += pix_val

        if pol:
            # Clip image representation to [-1, 1] for simplicity.
            frame = np.clip(frame, -1, 1)

        return frame

    def create_frame_stack(self, data, frames_stack, pol=False, tc=False):
        for b in range(data.shape[0]):
            frames_stack[b] = self.create_frame(data[b], pol, tc).ravel()

    def compare_frames(self, events, recon):
        """
        Function to plot expected event frame and reconstructed event frame
        """
        event_frame = events.reshape((self.W + 1, self.H + 1, 1)).detach().cpu().numpy()

        if self.decoder == "stream":
            recon_frame = self.create_frame(recon)
        elif self.decoder == "image":
            recon_frame = (
                recon.reshape((self.W + 1, self.H + 1, 1)).detach().cpu().numpy()
            )

        self.ax1.imshow(event_frame[:, :, 0])
        self.ax2.imshow(recon_frame[:, :, 0])

        return self.fig
