import gym
import airgym

import numpy as np
import matplotlib.pyplot as plt
import time
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback
import argparse


class EventRL:
    def __init__(self, args):
        """
        Create an event sim + AirSim gym env

        args:

        obs_type (str)      : event_stream (representation), or event_img (event frame)
        step_length (float) : Distance traveled by drone at each step (action)
        stack (int)         : Number of observations to stack
        data_len (int)      : 2 for X and Y, 3 to include polarity as well
        tc (bool)           : True if temporal coding is included
        ls (int)            : Size of latent vector
        lane_num (int)      : Lane number of choice according to AirSim environment
        image_shape (list)  : Resolutionxchannels of image
        goal (list)         : Coordinates of 'goal' in the obstacle track
        rep_weights (str)   : Path to trained weights for eVAE
        debug (bool)        : Setting this to True will visualize input event stream and reconstructed image

        """
        self.train_env = DummyVecEnv(
            [
                lambda: Monitor(
                    gym.make(
                        "airgym:airsim-event-v0",
                        ip_address="127.0.0.1",
                        obs_type=args.obs_type,
                        step_length=0.1,
                        stack=3,
                        data_len=args.data_len,
                        tc=args.tc,
                        ls=args.ls,
                        lane_num=2,
                        image_shape=(64, 64, 3),
                        goal=[0, -100, 0],
                        rep_weights=args.rep_weights,
                        debug=args.debug,
                    )
                )
            ]
        )

        # Evaluate on a harder obstacle course

        self.eval_env = DummyVecEnv(
            [
                lambda: Monitor(
                    gym.make(
                        "airgym:airsim-event-v0",
                        ip_address="127.0.0.1",
                        obs_type=args.obs_type,
                        step_length=0.1,
                        stack=3,
                        data_len=args.data_len,
                        tc=args.tc,
                        ls=args.ls,
                        lane_num=3,
                        image_shape=(64, 64, 3),
                        goal=[0, -100, 0],
                        rep_weights=args.rep_weights,
                        debug=args.debug,
                    )
                )
            ]
        )

        # env.render()

        # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
        tb_log_dir = args.output_dir

        if args.obs_type == "event_stream":
            model_type = "MlpPolicy"
        else:
            model_type = "CnnPolicy"

        self.model = PPO(
            model_type,
            self.train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=1,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gae_lambda=args.gae_lambda,
            gamma=args.gamma,
            device="cuda",
            tensorboard_log=tb_log_dir,
        )

        callbacks = []
        eval_callback = EvalCallback(
            self.eval_env,
            callback_on_new_best=None,
            n_eval_episodes=5,
            best_model_save_path=".",
            log_path=".",
            eval_freq=10000,
        )
        callbacks.append(eval_callback)

        self.kwargs = {}
        self.kwargs["callback"] = callbacks

    def train(self):
        self.model.learn(
            total_timesteps=5e5,
            tb_log_name="brp_run_" + str(time.time()),
            **self.kwargs
        )
        self.model.save("brp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--obs_type", type=str, default="event_stream")
    parser.add_argument("--data_len", type=int, default=2)
    parser.add_argument("--tc", action="store_true")
    parser.add_argument(
        "--rep_weights",
        type=str,
        default=None,
        required=True,
        help="pretrained representation",
    )
    parser.add_argument("--ls", type=int, default=8)

    # RL training hyperparams
    parser.add_argument("--n_steps", type=int, default=2048, help="n steps")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=4, help="n epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate"
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae_lambda")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    time.sleep(10)

    trainer = EventRL(args)
    trainer.train()
