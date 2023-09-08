
from tqdm import tqdm
import torch
import random
from matplotlib import pyplot as plt
import os
import numpy as np
from stable_baselines3 import A2C
from env import CustomEnv
from utils.load_data import load_data
import gymnasium as gym

# from gym_anytrading import gym_anytrading

df = load_data('csv_clean_5m/AAPL.csv')

window_size = 25
# df = df.iloc[:110]

env = CustomEnv(
    df=df,
    windows=window_size,
    # render_mode="human",
)

print("observation_space:", env.observation_space)

agent_file_name = "_new_model"

# %%

tensorboard_log = 'tensorboard_log'

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(tensorboard_log):
    os.makedirs(tensorboard_log)

env.reset(seed=42)

if 'model' in locals():
    del model

model = A2C('MlpPolicy', env, verbose=0,
            tensorboard_log=tensorboard_log)

loadfile = False
if loadfile:
    try:
        model = A2C.load(f"{agent_file_name}", env=env)
        print('model loaded successfully')
    except Exception as e:
        print('Fail to load model')

# Train the agent
total_timesteps = 1000
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the trained model with the current date in the filename
model.save(agent_file_name)

# model.get_env().unwrapped.env_method('render_all')

# %%
# reproduce training and test

df2 = df.iloc[:100]
env = CustomEnv(
    df=df2,
    windows=window_size,
    # render_mode="human",
)

env.reset(seed=42)

if 'model' in locals():
    del model

model = A2C.load(f"{agent_file_name}", env=env)
vec_env = model.get_env()

print('-' * 80)
seed = 42
# obs = env.reset(seed=seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

total_num_episodes = 10
tbar = tqdm(range(total_num_episodes))

for episode in tbar:
    obs = vec_env.reset()

    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)

        tbar.set_description(f'Episode: {episode} {reward}')
        tbar.update()

        total_reward += reward
        if done:
            break

    vec_env.unwrapped.env_method('render')

tbar.close()

# %%
