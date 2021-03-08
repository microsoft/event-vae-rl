import torch
import pandas as pd
import numpy as np
from copy import deepcopy
import random


class EventStreamArray:
    """
    Custom class to host event data as a "dataset"
    Contains timestamp, x, y, and polarity values
    """

    def __init__(self, csv_file, bn, bs, dl):
        self.filename = csv_file
        self.events = []
        with open(self.filename) as file:
            file.readline()
            file.readline()
            for l in file:
                event = l.split(" ")
                # Ignore the blank lines - need to correct this in source file
                if len(event) == 1:
                    continue
                # x, y, t, p
                self.events.append(
                    [float(event[2]), int(event[0]), int(event[1]), int(event[3])]
                )

        self.batch_num = bn
        self.batch_size = bs
        self.data_length = dl

    def get_event_batch(self, start_idx):
        """
        Return a batch of events of a given number dictated by batch_size
        """
        event_stack = []
        idx = start_idx

        t_start = self.events[start_idx][0]
        t_final = self.events[start_idx + self.batch_size - 1][0]

        dt = t_final - t_start

        # Iterate over events for a window of dt
        while idx - start_idx < self.batch_size:
            e_curr = deepcopy(self.events[idx])

            event_stack.append(e_curr)
            t_relative = float(t_final - e_curr[0]) / dt
            event_stack[idx - start_idx][0] = t_relative

            idx += 1

        event_batch_np = np.asarray(event_stack, dtype=np.float32)
        return event_batch_np

    def get_event_timeslice(self, start_idx, dt_max=16000):
        """
        Return a batch of events of a given time window dictated by dt_max
        """
        event_stack = []
        idx = start_idx

        t_start = self.events[start_idx][0]
        t_final = self.events[start_idx + self.batch_size - 1][0]

        dt_curr = 0.0

        dt = t_final - t_start

        # Iterate over events for a window of dt
        while dt_curr < dt_max:
            e_curr = deepcopy(self.events[idx])
            t_curr = e_curr[0]
            dt_curr = e_curr[0] - t_start

            event_stack.append(e_curr)

            idx += 1

        for event in event_stack:
            event[0] = (t_curr - event[0]) / dt_max

        event_batch_np = np.asarray(event_stack, dtype=np.float32)
        return event_batch_np

    def get_event_stack(self, event_np_stack):
        """
        Create a stack of event batches
        """
        for b in range(self.batch_num):
            start_idx = random.randint(0, len(self.events) - self.batch_size)
            event_np_stack[b] = self.get_event_batch(start_idx)

        return event_np_stack

    def extract(self, event_data):
        """
        Split event data into spatial and temporal parts and return separately
        """
        timestamps = torch.from_numpy(event_data[:, :, 0]).reshape(
            self.batch_num, 1, event_data.shape[1]
        )
        events_no_t = event_data[:, :, 1 : self.data_length + 1]

        event_data = torch.from_numpy(events_no_t).reshape(
            self.batch_num, event_data.shape[1], self.data_length
        )

        return event_data, timestamps
