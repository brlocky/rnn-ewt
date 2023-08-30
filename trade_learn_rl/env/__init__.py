from .custom_env import CustomEnv

from gymnasium.envs.registration import register
from copy import deepcopy

from gym_anytrading import datasets

from .portfolio import Portfolio, Trade, Portfolio, TradeDirection

register(
    id='custom-v0',
    entry_point='env:CustomEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)
