import numpy as np
import airsim

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete

from collections import OrderedDict
import time

from event_sim import EventSimulator
from event_processor import EventProcessor
import cv2
import torch
import random


"""
Gym env for AirSim combined with event simulation.
Loosely follows the official gym wrapper at 
https://github.com/microsoft/AirSim/tree/master/PythonClient/reinforcement_learning/airgym/
"""


class EvAirSim(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, image_shape, obs_type, stack, ls):

        # Choose event stream or event image as observation

        if obs_type == "event_stream":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=[1, stack * ls], dtype=np.uint8
            )
        elif obs_type == "event_img":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=image_shape, dtype=np.uint8
            )

        self._seed()

        self.viewer = None
        self.steps = 0
        self.no_episode = 0
        self.reward_sum = 0

    def __del__(self):
        raise NotImplementedError()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_reward(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def render(self, mode="human"):
        img = self._get_obs()
        if mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == "rgb_array":
            return img


class EvAirSimDrone(EvAirSim):
    def __init__(
        self,
        ip_address,
        obs_type,
        step_length,
        stack,
        data_len,
        tc,
        ls,
        image_shape,
        goal,
        lane_num,
        rep_weights,
        debug,
        noise=0.0,
        sparsity=0.0,
    ):
        super().__init__(image_shape, obs_type, stack, ls)

        self.step_length = step_length
        self.image_shape = image_shape
        self.goal = airsim.Vector3r(goal[0], goal[1], goal[2])
        self.start_ts = 0

        # Uses only discrete actions for now to allow for image-image event computations
        # at desired emulation of control frequency. Sim is essentially 'paused' between steps.

        self.action_space = spaces.Discrete(3)
        self.control_type = "position"
        self.state = {"position": np.zeros(3), "collision": False}

        # Initialize airsim client and do some map-specific initialization
        self.drone = airsim.MultirotorClient(ip=ip_address)

        self.pose = airsim.Pose()
        yaw = 270 * np.pi / 180
        pitch = 0
        roll = 0

        self.obs_type = obs_type

        # Change lane 1 limits to [-10, 10] for training env.

        if lane_num == 1:
            self.map_min = -25
            self.map_max = 25
        elif lane_num == 2:
            self.map_min = 20
            self.map_max = 40
        elif lane_num == 3:
            self.map_min = 50
            self.map_max = 70
        elif lane_num == 4:
            self.map_min = 80
            self.map_max = 100

        quat = airsim.to_quaternion(pitch, roll, yaw)
        self.pose.orientation = quat

        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )

        # Init event sim and event processor (events -> latent vector)
        self.eventsim = EventSimulator(image_shape[0], image_shape[1])
        self.eventproc = EventProcessor(
            data_len,
            image_shape,
            rep_weights,
            ls,
            tc,
            debug,
            noise_level=noise,
            sparsity=sparsity,
        )
        self.z = None
        self.init = True
        self.ls = ls
        self.stack = stack

        # Stack observations if needed. Currently uses stack of 3 observations

        if self.obs_type == "event_stream":
            self.obs_queue = [np.zeros([1, ls])] * stack
        elif self.obs_type == "event_img":
            self.obs_queue = [np.zeros([64, 64, 1])] * stack

        self.idx = 0

    def __del__(self):
        self.drone.reset()

    def computeZ(self, img, ts):
        with torch.no_grad():
            # cv2.imwrite(f"rgb_{self.idx}.png", img)
            # self.idx += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            img = cv2.add(img, 0.001)

            # Receive event image and list of events from simulator
            event_img, n_pix_ev, events = self.eventsim.image_callback(img, ts)
            if self.obs_type == "event_img":
                self.z = event_img.reshape([64, 64, 1]) * 255

            elif self.obs_type == "event_stream":
                bytestream = []

                if events is not None and events.shape[0] > 0:
                    bytestream = events.tolist()

                # Encode event list into a latent vector

                if len(bytestream) > 0:
                    self.z = (
                        self.eventproc.convertStream(bytestream, ts, n_pix_ev)
                        .cpu()
                        .numpy()
                    )
                else:
                    self.z = np.zeros([1, self.ls])

        return self.z

    def _setup_flight(self):
        # self.drone.reset()
        # self.drone.enableApiControl(True)
        self.pose.position.x_val = random.uniform(self.map_min, self.map_max)
        self.pose.position.y_val = 10
        self.pose.position.z_val = -5
        self.drone.simSetVehiclePose(self.pose, True)
        # self.drone.armDisarm(True)

    def _get_obs(self):

        # Get current image and timestamp from the drone in AirSim

        response = self.drone.simGetImages([self.image_request])
        while response[0].height == 0 or response[0].width == 0:
            response = self.drone.simGetImages([self.image_request])

        ts = time.time_ns()

        if self.init:
            self.start_ts = ts
            self.init = False

        image = np.reshape(
            np.fromstring(response[0].image_data_uint8, dtype=np.uint8),
            self.image_shape,
        )

        # Record drone state and collision data
        _drone_state = self.drone.getMultirotorState()
        position = _drone_state.kinematics_estimated.position.to_numpy_array()
        collision = self.drone.simGetCollisionInfo().has_collided

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = position
        self.state["collision"] = collision

        # Compute events between curr and prev image, corresponding 'observation'
        obs_new = self.computeZ(image, (ts - self.start_ts))

        # Push newest obs into obs queue
        self.obs_queue.pop(0)
        self.obs_queue.append(obs_new)

        # self.start_ts = ts

        if self.obs_type == "event_stream":
            obs = np.hstack(self.obs_queue)
        elif self.obs_type == "event_img":
            obs = np.dstack(self.obs_queue)

        return obs

    def _compute_reward(self):
        pos = self.state["position"]
        prev_pos = self.state["prev_position"]

        reward = 0
        done = False

        # +100 if drone has reached the end of the track (~100-105 meters)

        if abs(pos[1]) >= 105:
            done = True
            reward = 100
            print(f"{abs(pos[1])},")
            return reward, done

        # -100 if drone has collided with anything

        elif self.state["collision"] == True:
            done = True
            reward = -100
            print(f"{abs(pos[1])},")
            return reward, done

        # Incremental rewards based on Y axis distance traveled

        reward += abs(pos[1]) - abs(prev_pos[1])
        return reward, done

    def _do_action(self, action):
        action = self.actions_to_op(action)

        pos = self.state["position"]
        self.pose.position.x_val = float(pos[0] + action[0])
        self.pose.position.y_val = float(pos[1] + action[1])
        self.pose.position.z_val = float(pos[2])
        self.drone.simSetVehiclePose(self.pose, False)

    def forward(self):
        return [0.0, -self.step_length, 0.0]

    def right(self):
        return [0.707 * self.step_length, -0.707 * self.step_length, 0.0]

    def left(self):
        return [-0.707 * self.step_length, -0.707 * self.step_length, 0.0]

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self.obs_queue = [np.zeros([1, self.ls])] * self.stack

        return self._get_obs()

    def actions_to_op(self, action):
        switcher = {
            0: self.forward,
            1: self.right,
            2: self.left,
        }

        func = switcher.get(action, lambda: "Invalid Action!")
        return func()
