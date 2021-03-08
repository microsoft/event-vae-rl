from gym.envs.registration import register

register(
    id='airsim-event-v0',
    entry_point='airgym.envs:EvAirSimDrone',
)
