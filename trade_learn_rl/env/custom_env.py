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
    Sell = 0
    Hold = 1
    Close = 2
    Buy = 3


class Positions(Enum):
    NoPosition = 0
    Short = 1
    Long = 2


class CustomEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, render_mode=None):
        super().__init__()
        assert window_size >= 1
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Data
        self.df = df
        self.window_size = window_size
        self.prices, self.dates, self.volumes, self.signal_features = self._process_data()

        # Settings
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        self._initial_balance = 10000
        self._porfolio = Portfolio(self._initial_balance)
        self._share_size = 5

        # Rewards
        self._initial_reward_factor = 1.
        self._reward_factor_fraction = 0.01
        self._reward_factor = self._initial_reward_factor - self._reward_factor_fraction
        self._total_reward = 0.

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = 0
        self._position = Positions.NoPosition
        self._action = Actions.Hold

        # History
        self._history = []
        self._reset_counter = -1

        observation = self._get_observation()
        # self.shape = (window_size, observation.shape[1])
        self.shape = observation.shape

        # self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_counter += 1

        self.prices, self.dates, self.volumes, self.signal_features = self._process_data()
        self._porfolio = Portfolio(self._initial_balance)
        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = 0
        self._position = Positions.NoPosition
        self._action = Actions.Hold

        self._total_reward = 0.
        self._reward_factor = self._initial_reward_factor - self._reward_factor_fraction
        # self._position_history = (self.window_size * [None]) + [self._position]

        # Create History
        info = self._get_history_info()
        history = (self.window_size * [info]) + [info]
        self._history.append(history)

        observation = self._get_observation()

        return observation, info

    def step(self, action):

        self._action = Actions(action)

        # We use the price where the action was taken
        current_price = self.prices[self._current_tick]

        # Open Long
        if self._action == Actions.Buy and self._porfolio.open_trade(
                TradeDirection.Long, current_price, self._share_size):
            self._position = Positions.Long
            self._last_trade_tick = self._current_tick

        # Open Short
        if self._action == Actions.Sell and self._porfolio.open_trade(
                TradeDirection.Short, current_price, self._share_size):
            self._position = Positions.Short
            self._last_trade_tick = self._current_tick

        # Close Position
        trade_pnl = 0.0
        if self._action == Actions.Close:
            trade_pnl = self._porfolio.close_trade(current_price)
            self._position = Positions.NoPosition

        self._porfolio.update(current_price)

        step_reward = self._calculate_reward(self._action, trade_pnl)
        self._total_reward += step_reward

        info = self._update_history(self._current_tick)

        self._current_tick += 1
        observation = self._get_observation()

        self._terminated = True if self._current_tick == self._end_tick else False

        return observation, step_reward, self._terminated, False, info

    def _calculate_reward(self, action, trade_pnl):
        """ if action == Actions.Buy:
            step_reward += 0.3

        if action == Actions.Sell:
            step_reward += 0.3
         """
        # open_pnl = self._porfolio.get_open_pnl()
        # total_pnl = self._porfolio.get_total_pnl()
        step_reward = 0.0

        last_position = self._history[self._reset_counter][-1]['position']
        trade_ticks = self._current_tick - self._last_trade_tick

        # Opened Position
        if (action == Actions.Buy or action == Actions.Sell) and last_position == Positions.NoPosition:
            step_reward += 10 * (self._initial_reward_factor - self._reward_factor)

        # Position Closed
        if action == Actions.Close:
            if (last_position == Positions.Long or last_position == Positions.Short):
                if trade_pnl >= 0:
                    step_reward = trade_pnl * (trade_ticks * self._reward_factor)
                else:
                    step_reward = trade_pnl * trade_ticks if trade_ticks > 3 else 0

                # Decrease reward factor on each close
                self._reward_factor -= self._reward_factor_fraction if self._reward_factor > self._reward_factor_fraction else self._reward_factor

            # Position Closed without trade
            else:
                step_reward = -5 * (self._initial_reward_factor - self._reward_factor)

        # Holding Position
        if action == Actions.Hold and (last_position == Positions.Long or last_position == Positions.Short):
            hold_pnl = self._porfolio.get_open_pnl()
            if hold_pnl >= 0:
                step_reward = hold_pnl * (trade_ticks * self._reward_factor)
            else:
                step_reward = hold_pnl * trade_ticks if trade_ticks > 3 else 0

        # Maybe decrease reward factor on errors ?

        return step_reward

    def _get_observation(self):
        info = self._get_history_info()
        # Update current position
        self.signal_features[self._current_tick - 1, -3] = info['position'].value

        # Update Porfolio Pnl
        self.signal_features[self._current_tick - 1, -2] = info['total_profit']

        # Update Porfolio value
        self.signal_features[self._current_tick - 1, -1] = info['balance']

        observation = self.signal_features[(
            self._current_tick - self.window_size):self._current_tick]

        # Price, Price diff, position, pnl, balance
        reshaped_observation = observation.reshape(observation.shape[0], 3, 3)

        return reshaped_observation

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        dates = self.df.index
        volumes = self.df.loc[:, 'Volume'].to_numpy()
        vwap = self.df.loc[:, 'feature_vwap'].to_numpy()
        ema = self.df.loc[:, 'feature_ema'].to_numpy()
        atr = self.df.loc[:, 'feature_atr'].to_numpy()
        rsi = self.df.loc[:, 'feature_rsi'].to_numpy()

        # Adding three empty columns
        input_features = np.zeros((len(dates), 3), dtype=np.float64)
        signal_features = np.hstack((
            prices.reshape(-1, 1),
            vwap.reshape(-1, 1),
            ema.reshape(-1, 1),
            atr.reshape(-1, 1),
            rsi.reshape(-1, 1),
            volumes.reshape(-1, 1),
            input_features))

        return prices, dates, volumes, signal_features

    def _get_history_info(self):
        return {
            "total_reward": self._total_reward,
            "balance": self._porfolio.get_balance(),
            "total_profit": self._porfolio.get_total_pnl(),
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

        # date_strings = [str(date) for date in self.dates]
        date_strings = [str(date) if (date.hour in [9, 12, 15, 18]) and date.minute == 0 else
                        str(i) for i, date in enumerate(self.dates)]

        ax1.plot(date_strings, self.prices)
        ax1.set_title('Price')

        # Use the modified date_strings
        ax1.set_xticklabels(date_strings, rotation=45, fontsize=6)

        render_hist = self._history[self._reset_counter - 1]
        short_ticks = []
        long_ticks = []
        close_ticks = []
        for i in range(self._start_tick, len(render_hist) - 1):
            position = render_hist[i]['position']
            action = render_hist[i]['action']
            if action == Actions.Buy:
                long_ticks.append(i)
            elif action == Actions.Sell:
                short_ticks.append(i)
            elif action == Actions.Close:
                close_ticks.append(i)

        plt.plot(short_ticks,
                 self.prices[short_ticks], 'ro', label='Short')
        plt.plot(long_ticks,
                 self.prices[long_ticks], 'go', label='Long')
        plt.plot(close_ticks,
                 self.prices[close_ticks], 'bx', label='Close')

        total_reward = render_hist[-1]['total_reward']
        total_profit = render_hist[-1]['total_profit']

        plt.suptitle(
            "Total Reward: %.6f" % total_reward + ' ~ '
            + "Total Profit: %.6f" % total_profit
        )
        plt.tight_layout()
        plt.show()

        # Rewards
        total_rewards = [obj['total_reward'] for obj in render_hist]
        fig3, ax3 = plt.subplots(figsize=(12, 2))
        ax3.plot(date_strings, total_rewards)
        ax3.set_title('Rewards')
        ax3.set_xticklabels(date_strings, rotation=0, fontsize=1)
        plt.tight_layout()
        plt.show()

        # Pnl
        total_profits = [obj['total_profit'] for obj in render_hist]
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(date_strings, total_profits)
        ax2.set_title('Pnl')
        ax2.set_xticklabels(date_strings, rotation=0, ha='right', fontsize=1)
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
