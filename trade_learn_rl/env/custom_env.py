import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


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

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        super().__init__()

        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert len(frame_bound) == 2
        self.frame_bound = frame_bound

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = False
        self._current_tick = 0
        self._last_trade_tick = 0
        self._position = Positions.NoPosition
        self._position_history = []
        self._shares_history = []
        self._total_reward = 0
        self._total_profit = 0
        self._first_rendering = True
        self.history = {}
        self._share_size = 10
        self._shares = 0
        self._initial_balance = 1000
        self._balance = 0

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = 0
        self._position = Positions.NoPosition
        self._position_history = (self.window_size * [None]) + [self._position]
        self._shares = 0
        self._shares_history = (self.window_size * [0]) + [self._shares]
        self._total_reward = 0.
        self._total_profit = 0.
        self._first_rendering = True
        self.history = {}
        self._balance = self._initial_balance

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._terminated = True

        action = Actions(action)

        # Open Long
        if action == Actions.Buy:
            self._position = Positions.Long
            self._last_trade_tick = self._current_tick
            self._shares = self._share_size

        # Open Short
        if action == Actions.Sell:
            self._position = Positions.Short
            self._last_trade_tick = self._current_tick
            self._shares = self._share_size

        # Close Position
        if action == Actions.Close:
            self._position = Positions.NoPosition
            self._shares = 0

        self._position_history.append(self._position)
        self._shares_history.append(self._shares)

        step_reward = self._calculate_reward(action)

        self._update_profit(action)

        self._total_reward += step_reward

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, self._terminated, False, info

    def _get_observation(self):
        currentPositon = self._position_history[self._current_tick].value
        currentShares = self._shares_history[self._current_tick]
        currentShares = self._shares_history[self._current_tick]

        # Update values only in the last tick of self.signal_features
        # Update current postion
        self.signal_features[self._current_tick, -3] = currentPositon

        # Update current shares
        self.signal_features[self._current_tick, -2] = currentShares

        # Update Porfolio value
        self.signal_features[self._current_tick, -1] = self._balance

        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.NoPosition:
                color = 'black'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ '
            + "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        noposition_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.NoPosition:
                noposition_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')
        plt.plot(noposition_ticks, self.prices[noposition_ticks], 'x')

        if title:
            plt.title(title)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ '
            + "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _calculate_reward(self, action):
        step_reward = 0
        if action == Actions.Buy:
            step_reward = 1

        if action == Actions.Sell:
            step_reward = 1

        if action == Actions.Close:
            step_reward = self._get_trade_pnl(self._last_trade_tick, self._current_tick)

        return step_reward

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        # Adding three empty columns
        empty_columns = np.empty((signal_features.shape[0], 3))
        signal_features_with_empty = np.hstack((signal_features, empty_columns))

        return prices, signal_features_with_empty

    def _get_trade_pnl(self, start_index, end_index):
        buy_price = self.prices[start_index]
        sell_price = self.prices[end_index]
        shares = self._shares_history[start_index]
        pnl = 0

        # Long Pnl
        if self._position_history[start_index] == Positions.Long:
            pnl = shares * (sell_price - buy_price)

        # Short Pnl
        if self._position_history[start_index] == Positions.Short:
            pnl = shares * (buy_price - sell_price)

        return pnl

    def _update_profit(self, action):
        self._total_profit = 0

        if self._position != Positions.NoPosition:
            self._total_profit = self._get_trade_pnl(
                self._current_tick - 1, self._current_tick)

        if action == Actions.Close:
            self._balance += self._get_trade_pnl(self._last_trade_tick,
                                                 self._current_tick)

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
