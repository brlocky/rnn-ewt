import os
from re import T
import numpy as np
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from torch import nn

from env.custom_env import CustomEnv

selected_model = RecurrentPPO


def create_env(df):
    window_size = 100
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

    # policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[512, 512]))
    policy_kwargs = dict(
        log_std_init=0.0,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256], vf=[256]),
        n_lstm_layers=1,
        lstm_hidden_size=512,
        shared_lstm=False,
        enable_critic_lstm=True,
        # normalize_images=False,
    )
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=2,
        tensorboard_log=tensorboard_log,
        seed=seed,
        batch_size=256,  # [8, 16, 32, 64, 128, 256, 512]
        n_steps=1024,  # [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        gamma=0.99,  # [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
        # learning_rate=0.001,
        learning_rate=5e-4,
        ent_coef=0.005,
        clip_range=0.1,  # [0.1, 0.2, 0.3, 0.4]
        n_epochs=10,  # [1, 5, 10, 20]
        gae_lambda=0.98,  # [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
        max_grad_norm=0.9,  # [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
        vf_coef=0.19,
        policy_kwargs=policy_kwargs,
        target_kl=0.08,
        normalize_advantage=True,
    )

    """     normalize: true
    n_envs: 32
    n_timesteps: !!float 10e7
    policy: 'MlpLstmPolicy'
    n_steps: 256
    batch_size: 256
    gae_lambda: 0.95
    gamma: 0.999
    n_epochs: 10
    ent_coef: 0.001
    learning_rate: lin_3e-4
    clip_range: lin_0.2
    policy_kwargs: "dict(
                        ortho_init=False,
                        activation_fn=nn.ReLU,
                        lstm_hidden_size=64,
                        enable_critic_lstm=True,
                        net_arch=dict(pi=[64], vf=[64])
                    )" """

    return model


def load_model(model_name: str, env: gym.Env):
    return selected_model.load(model_name, env=env)


def train_model(model: RecurrentPPO, total_timesteps=10_000, progress_bar=False):
    return model.learn(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=HParamCallback(),
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
