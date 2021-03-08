import gym
import airgym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# evae_weights = "weights/evae_full_tc.pt"
evae_weights = "weights/evae_xy.pt"

policy_weights = "trained/brp_xy"

rep_type = "brp_xy"

tc = False
data_len = 2

if rep_type == "brp_full":
    tc = True
    data_len = 3

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
                stack=3,
                data_len=data_len,
                tc=tc,
                ls=8,
                lane_num=1,
                image_shape=(64, 64, 3),
                goal=[0, -100, 0],
                rep_weights=evae_weights,
                debug=False,
                noise=0.0,
                sparsity=0.2,
            )
        )
    ]
)

model = PPO.load(policy_weights)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, return_episode_rewards=True
)

print(mean_reward)
print(std_reward)

obs = env.reset()

