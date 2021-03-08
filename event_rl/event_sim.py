import numpy as np
from types import SimpleNamespace
from numba import njit, prange, set_num_threads
import airsim
import time
from datetime import datetime
import cv2
import glob
import matplotlib.pyplot as plt

EVENT_TYPE = np.dtype(
    [("timestamp", "f8"), ("x", "u2"), ("y", "u2"), ("polarity", "b")], align=True
)

TOL = 0.2
MINIMUM_CONTRAST_THRESHOLD = 0.1

# Event cam related parameters were chosen in order to allow for fast realtime operation
# in tandem with AirSim. Can be changed as needed.

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_us": 500,
        "max_events_per_frame": 50000,
    }
)

# Loosely follows official implementation in AirSim
# https://github.com/microsoft/AirSim/blob/master/PythonClient/eventcamera_sim/event_simulator.py


@njit
def esim(
    x_start,
    x_end,
    current_image,
    previous_image,
    delta_time,
    crossings,
    last_time,
    output_events,
    spikes,
    refractory_period_us,
    max_events_per_frame,
    n_pix_row,
):
    count = 0

    # 5000 us = 5 ms to "emulate" 200 Hz capture of images
    maxSpikes = int(5000 / refractory_period_us)
    for x in range(x_start, x_end):
        itdt = np.log(current_image[x])
        it = np.log(previous_image[x])
        deltaL = itdt - it

        if np.abs(deltaL) < TOL:
            continue

        pol = np.sign(deltaL)

        crossUpdate = pol * TOL
        crossings[x] = np.log(crossings[x]) + crossUpdate

        lb = crossings[x] - it
        ub = crossings[x] - itdt

        posCheck = lb > 0 and (pol == 1) and ub < 0
        negCheck = lb < 0 and (pol == -1) and ub > 0

        spikeNums = (itdt - crossings[x]) / TOL
        crossCheck = posCheck + negCheck
        spikeNums = np.abs(int(spikeNums * crossCheck))

        crossings[x] = itdt - crossUpdate

        # spikes contains an 'event image' representation
        if np.abs(spikeNums) > 0.0001:
            spikes[x] = 255 if spikeNums > 0 else 125

        spikeNums = maxSpikes if spikeNums > maxSpikes else spikeNums

        current_time = last_time
        for i in range(spikeNums):
            output_events[count].x = x // n_pix_row
            output_events[count].y = x % n_pix_row
            output_events[count].timestamp = np.round(current_time * 1e-3, 6)
            output_events[count].polarity = 1 if pol > 0 else -1

            count += 1
            current_time += (delta_time) / spikeNums

            # max_events_per_frame needs to be tuned according to resolution
            if count == max_events_per_frame:
                return count

    return count


class EventSimulator:
    def __init__(self, H, W, first_image=None, first_time=None, config=CONFIG):
        self.H = H
        self.W = W
        self.config = config
        self.last_image = None
        if first_image is not None:
            assert first_time is not None
            self.init(first_image, first_time)

        self.npix = H * W

    def init(self, first_image, first_time):
        print("Initialized event camera simulator with sensor size:", first_image.shape)
        print(
            "and contrast thresholds: C-=",
            self.config.contrast_thresholds[0],
            "C+=",
            self.config.contrast_thresholds[1],
        )

        self.resolution = first_image.shape  # The resolution of the image

        # We ignore the 2D nature of the problem as it is not relevant here
        # It makes multi-core processing more straightforward
        first_image = first_image.reshape(-1)

        self.last_image = first_image.copy()

        # Buffer for current image
        self.current_image = first_image.copy()

        self.last_event_timestamps = np.full(
            first_image.shape, -np.inf, dtype="float64"
        )
        self.last_time = first_time

        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.event_count = 0
        self.n_pix_ev = 0
        self.spikes = np.zeros((self.npix))

    def image_callback(self, new_image, new_time):
        if self.last_image is None:
            self.init(new_image, new_time)
            return self.spikes, 0, None

        assert new_time > 0
        assert new_image.shape == self.resolution
        new_image = new_image.reshape(-1)  # Free operation

        # Copy is faster than reallocating memory
        np.copyto(self.current_image, new_image)

        delta_time = 5000  # new_time - self.last_time

        config = self.config
        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.spikes = np.zeros((self.npix))

        self.crossings = self.last_image.copy()
        self.n_pix_ev = 0
        self.event_count = esim(
            0,
            self.current_image.size,
            self.current_image,
            self.last_image,
            delta_time,
            self.crossings,
            self.last_time,
            self.output_events,
            self.spikes,
            config.refractory_period_us,
            config.max_events_per_frame,
            self.W,
        )
        self.n_pix_ev = len(np.nonzero(self.spikes)[0])

        np.copyto(self.last_image, self.current_image)
        self.last_time = new_time

        # Sort events according to time stamps and output both stream and image
        result = self.output_events[: self.event_count]
        result.sort(order=["timestamp"], axis=0)
        return self.spikes, self.n_pix_ev, result


if __name__ == "__main__":
    pass
