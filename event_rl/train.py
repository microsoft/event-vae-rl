import gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback

weights_file = "weights/evae_tc_phase.pt"

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
noise (float)       : Percentage of pixels that could fire randomly even without activity
sparsity (float)    : Percentage of pixels that could refuse to fire even under activity

"""

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-event-v0",
                ip_address="127.0.0.1",
                obs_type="event_stream",
                step_length=0.1,
                data_len=3,
                ls=8,
                lane_num=2,
                image_shape=(64, 64, 3),
                goal=[0, -100, 0],
                rep_weights=weights_file,
                debug=False,
            )
        )
    ]
)

eval_env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-event-v0",
                ip_address="127.0.0.1",
                obs_type="event_stream",
                step_length=0.1,
                data_len=3,
                ls=8,
                lane_num=3,
                image_shape=(64, 64, 3),
                goal=[0, -100, 0],
                rep_weights=weights_file,
                debug=False,
            )
        )
    ]
)

# env.render()
episodes = 10
episode_rewards = []


model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=32,
    verbose=1,
    n_epochs=8,
    learning_rate=1e-4,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

callbacks = []
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

model.learn(total_timesteps=5e5, tb_log_name="ppo_run_" + str(time.time()), **kwargs)
model.save("brp_full")

