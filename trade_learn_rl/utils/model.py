import os
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
import gymnasium as gym
from env.custom_env import CustomEnv


def create_model(env: gym.Env):
    tensorboard_log = 'tensorboard_log'
    seed = np.random.seed()

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    # model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log, n_steps=25, seed=seed)

    model = A2C('MlpPolicy',
                env,
                verbose=0,
                tensorboard_log=tensorboard_log,
                seed=seed,
                policy_kwargs=dict(net_arch=[512, 512]),
                normalize_advantage=True,
                learning_rate=0.0001
                )

    # model = PPO('MlpPolicy', env, verbose=0,tensorboard_log=tensorboard_log, n_steps=25, seed=seed)
    return model


def create_env(df):
    window_size = 10

    env = CustomEnv(
        df=df,
        window_size=window_size,
    )

    return env
