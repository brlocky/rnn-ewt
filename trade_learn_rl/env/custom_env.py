import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
from .portfolio import Portfolio, TradeDirection
import plotly.graph_objs as go


class Actions(Enum):
    NoAction = 0
    Buy = 1
    Sell = 2
    Close = 3


class Positions(Enum):
    NoPosition = 0
    Long = 1
    Short = 2


class CustomEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, render_mode=None):
        super().__init__()
        assert window_size >= 1
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        self.df = df
        self.window_size = window_size
        self.prices, self.dates, self.volumes, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self._initial_balance = 10000

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = False

        self._current_tick = self._start_tick
        self._position = Positions.NoPosition
        self._action = Actions.NoAction

        self._total_reward = 0

        self._porfolio = Portfolio(self._initial_balance)
        self._share_size = 1

        self._history = []
        self._reset_counter = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_counter += 1

        self.prices, self.dates, self.volumes, self.signal_features = self._process_data()
        self._porfolio = Portfolio(self._initial_balance)
        self._terminated = False
        self._current_tick = self._start_tick
        self._position = Positions.NoPosition
        self._action = Actions.NoAction

        self._total_reward = 0.
        # self._position_history = (self.window_size * [None]) + [self._position]

        info = self._get_history_info()
        observation = self._get_observation()

        # Create History
        info = self._get_history_info()
        history = (self.window_size * [info])
        self._history.append(history)

        return observation, info

    def step(self, action):

        self._action = Actions(action)

        # We use the price where the action was taken
        current_price = self.prices[self._current_tick]

        # Open Long
        if self._action == Actions.Buy:
            self._porfolio.open_trade(
                TradeDirection.Long, current_price, self._share_size)
            self._position = Positions.Long

        # Open Short
        if self._action == Actions.Sell:
            self._porfolio.open_trade(
                TradeDirection.Short, current_price, self._share_size)
            self._position = Positions.Short

        # Close Position
        if self._action == Actions.Close:
            self._porfolio.close_trade(current_price)
            self._position = Positions.NoPosition

        self._porfolio.update(current_price)

        step_reward = self._calculate_reward(self._action)
        self._total_reward += step_reward

        observation = self._get_observation()
        info = self._update_history(self._current_tick)

        self._current_tick += 1
        self._terminated = True if self._current_tick == self._end_tick else False

        return observation, step_reward, self._terminated, False, info

    def _calculate_reward(self, action):
        step_reward = self._porfolio.get_total_pnl()

        """ if action == Actions.Buy:
            step_reward += 0.3

        if action == Actions.Sell:
            step_reward += 0.3
         """
        last_position = self._history[self._reset_counter][-1]['position']
        if action == Actions.Buy and last_position != Positions.NoPosition:
            step_reward -= 0.3

        if action == Actions.Sell and last_position != Positions.NoPosition:
            step_reward -= 0.3

        if action == Actions.Close and last_position != Positions.NoPosition:
            step_reward -= 0.5

        if action == Actions.NoAction and last_position == Positions.NoPosition:
            step_reward -= 0.1

        return step_reward

    def _get_observation(self):
        info = self._get_history_info()
        # Update current position
        self.signal_features[self._current_tick, -3] = info['position'].value

        # Update current shares
        self.signal_features[self._current_tick, -2] = self._share_size

        # Update Porfolio value
        self.signal_features[self._current_tick, -1] = info['total_profit']

        observation = self.signal_features[(
            self._current_tick - self.window_size + 1):self._current_tick + 1]
        return observation

    def _get_history_info(self):
        return {
            "total_reward": self._total_reward,
            "total_profit": self._porfolio.get_balance(),
            "position": self._position,
            "action": self._action,
        }

    def _update_history(self, index):
        info = self._get_history_info()
        self._history[self._reset_counter].append(info)

        # Return history
        return info

    def render(self, mode='human'):
        # Create a new figure for Price
        fig1, ax1 = plt.subplots(figsize=(12, 3))
        ax1.plot(self.dates, self.prices)
        ax1.set_title('Price')

        render_hist = self._history[self._reset_counter - 1]
        short_ticks = []
        long_ticks = []
        close_ticks = []
        for i in range(self._start_tick, len(render_hist) - 1):
            position = render_hist[i]['position']
            action = render_hist[i]['action']
            if position == Positions.Short:
                short_ticks.append(i)
            elif position == Positions.Long:
                long_ticks.append(i)

            if action == Actions.Close:
                close_ticks.append(i)

        plt.plot(self.dates[short_ticks],
                 self.prices[short_ticks], 'ro', label='Short')
        plt.plot(self.dates[long_ticks],
                 self.prices[long_ticks], 'go', label='Long')
        plt.plot(self.dates[close_ticks],
                 self.prices[close_ticks], 'bx', label='Close')

        total_reward = render_hist[-1]['total_reward']
        total_profit = render_hist[-1]['total_profit']

        plt.suptitle(
            "Total Reward: %.6f" % total_reward + ' ~ '
            + "Total Profit: %.6f" % total_profit
        )

        plt.tight_layout()
        plt.show()

        # Create a new figure for Volume
        """ fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(self.dates, self.volumes)
        ax2.set_title('Volume')

        plt.tight_layout()
        plt.show() """

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        dates = self.df.index
        volumes = self.df.loc[:, 'Volume'].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        # Adding three empty columns
        empty_columns = np.zeros((signal_features.shape[0], 3))
        signal_features_with_empty = np.hstack((signal_features, empty_columns))

        return prices, dates, volumes, signal_features_with_empty

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
