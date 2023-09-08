import os
import numpy as np
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from env.custom_env import CustomEnv

selected_model = RecurrentPPO


def create_env(df):
    window_size = 20
    lstm_window = 10

    env = CustomEnv(
        df=df,
        windows=window_size,
        lstm_window=lstm_window,
    )

    return env


def create_model(env: gym.Env):
    tensorboard_log = 'tensorboard_log'
    seed = np.random.seed()

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    # model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log, n_steps=25, seed=seed)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[512, 512]))
    model = selected_model(
        "MlpLstmPolicy",
        env,
        verbose=0,
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs=policy_kwargs
    )

    return model


def load_model(model_name: str, env: gym.Env):
    return selected_model.load(model_name, env=env)


def train_model(model: RecurrentPPO, total_timesteps=10_000):
    return model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=HParamCallback()
    )


class HParamCallback(BaseCallback):
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "train/reward": 0.0,
            "train/balance": 0.0,
            "train/total_profit": 0.0,
            "train/open_pnl": 0.0,
            "train/total_reward": 0.0,
            "train/action": 0.0,
            "train/position": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        infos = self.locals['infos'][-1]
        # Log scalar value (here a random variable)
        self.logger.record('train/balance', infos['balance'])
        self.logger.record('train/total_profit', infos['total_profit'])
        self.logger.record('train/open_pnl', infos['open_pnl'])
        self.logger.record('train/total_reward', infos['total_reward'])
        self.logger.record('train/action', infos['action'].value)
        self.logger.record('train/position', infos['position'].value)

        return True
