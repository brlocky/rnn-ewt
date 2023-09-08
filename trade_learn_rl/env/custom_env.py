from collections import Counter
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from .portfolio import Portfolio, TradeDirection
from sklearn.preprocessing import MinMaxScaler


class Actions(Enum):
    NoAction = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    NoPosition = 0
    Short = 1
    Long = 2


class CustomEnv(gym.Env):

    def __init__(self, df, windows, lstm_window):
        super().__init__()
        assert windows >= 1
        assert windows > lstm_window
        assert df.ndim == 2

        self.windows = windows
        self.lstm_window = lstm_window
        self.done = False

        # Trading Settings
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        self._initial_balance = 1000
        self._share_size = 1
        self._porfolio = Portfolio(self._initial_balance)

        # episode
        self._start_tick = self.windows
        self._current_tick = self._start_tick
        self._last_trade_open_tick = 0
        self._last_trade_close_tick = 0
        self._position = Positions.NoPosition
        self._action = Actions.NoAction
        self._total_reward = 0.0

        # Data
        self.sc = MinMaxScaler(feature_range=(0, 1))
        self.df = df
        self.features = self._process_data()
        self._end_tick = len(self.prices_close) - 1

        # History
        self._all_history = []
        self._reset_counter = -1

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            # shape=[
            #    self.windows - self.lstm_window,
            #    self.lstm_window,
            #    len(self.features[0])
            # ] """
            shape=[
                self.windows,
                len(self.features[0])
            ]
        )

        # ValueError: could not broadcast input array from shape (20,10,7) into shape (10,30,7)
        # (n_samples x timesteps x n_features)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_counter += 1

        # Reset variables
        self.done = False
        self._total_reward = 0.0
        self._position = Positions.NoPosition
        self._action = Actions.NoAction
        self._last_trade_open_tick = 0
        self._last_trade_close_tick = 0
        self._current_tick = self._start_tick

        # Refresh main variables
        self.features = self._process_data()

        # Refresh Portfolio
        self._porfolio = Portfolio(self._initial_balance)

        # Create History
        info = self._get_step_info()
        history = (self.windows * [info]) + [info]
        self._all_history.append(history)

        observation = self._get_observation()

        return observation, info

    def step(self, action):
        self._current_tick += 1

        last_position = self._position

        # Once this should be a new candlestick we will use the open price
        current_price = self.prices_open[self._current_tick]
        self._porfolio.update(current_price)

        self._action = Actions(action)
        # Open Long
        if self._action == Actions.Buy:
            if self._porfolio._current_trade and self._porfolio._current_trade.trade_direction == TradeDirection.Short:
                self._porfolio.close_trade(current_price)
                self._position = Positions.NoPosition
                self._last_trade_close_tick = self._current_tick

            if self._porfolio.open_trade(
                    TradeDirection.Long, current_price, self._share_size):
                self._position = Positions.Long
                self._last_trade_open_tick = self._current_tick

        # Open Short
        if self._action == Actions.Sell:
            if self._porfolio._current_trade and self._porfolio._current_trade.trade_direction == TradeDirection.Long:
                self._porfolio.close_trade(current_price)
                self._position = Positions.NoPosition
                self._last_trade_close_tick = self._current_tick

            if self._porfolio.open_trade(
                    TradeDirection.Short, current_price, self._share_size):
                self._position = Positions.Short
                self._last_trade_open_tick = self._current_tick

        # Close Position
        if self._action == Actions.NoAction:
            if self._porfolio.close_trade(current_price):
                self._position = Positions.NoPosition
                self._last_trade_close_tick = self._current_tick

        step_reward = self._calculate_reward()

        self._total_reward += step_reward

        # Update History
        info = self._get_step_info()
        self._all_history[self._reset_counter].append(info)

        observation = self._get_observation()

        if self._current_tick == self._end_tick or self._porfolio.get_balance() <= 0:
            self.done = True

        return observation, step_reward, self.done, False, info

    def _smooth_reward(self, value: float, trade_ticks: int, smoothing_coefficient=0.3):
        return (1 + np.log(trade_ticks) * smoothing_coefficient) * value

    def _is_price_increase(self):
        len_to_check = 3

        if self._current_tick + len_to_check >= len(self.prices_close):
            return 0

        if self.prices_close[self._current_tick] < self.prices_close[self._current_tick + len_to_check]:
            return 1
        else:
            return -1

    def _calculate_reward(self):
        # open_pnl = self._porfolio.get_open_pnl()
        # total_pnl = self._porfolio.get_total_pnl()
        step_reward = 0.0
        last_open_position_ticks = self._current_tick - self._last_trade_open_tick
        last_close_position_ticks = self._current_tick - self._last_trade_close_tick

        last_pnl = self._get_history("open_pnl", -1)
        last_position = self._get_history("position", -1)
        current_balance = self._porfolio.get_balance()
        price_will_increase = self._is_price_increase()

        close_pivots = self.pivots[self._current_tick - 8:self._current_tick][0]
        pivot_high = np.any(close_pivots == 1)
        pivot_low = np.any(close_pivots == -1)

        rsi_values = self.rsi[self._current_tick - 8:self._current_tick]
        rsi_high = np.any(rsi_values > 70)
        rsi_low = np.any(rsi_values < 30)

        # New Long
        if self._action == Actions.Buy:
            """ if pivot_low:
                step_reward += 0.5
            else:
                step_reward -= 0.3 """

            if rsi_low:
                step_reward += 0.5
            else:
                step_reward -= 0.3

        # New Short
        if self._action == Actions.Sell:
            """ if pivot_high:
                step_reward += 0.5
            else:
                step_reward -= 0.3 """

            if rsi_high:
                step_reward += 0.5
            else:
                step_reward -= 0.3

        # Position Holding
        if (self._action == Actions.Buy or self._action == Actions.Sell):
            balance = self._porfolio.get_balance()
            # last_balance = self._get_history("balance", -1)
            last_balance = self._get_history("balance", -last_open_position_ticks)
            step_reward += (balance - (last_balance)) / last_balance * 100

        # Position Closed
        if self._action == Actions.NoAction and (last_position == Positions.Long or last_position == Positions.Short):
            balance = self._porfolio.get_balance()
            last_balance = self._get_history("balance", -last_open_position_ticks)
            step_reward += (balance - (last_balance)) / last_balance * 100

        if self._action == Actions.NoAction and last_position == Positions.NoPosition and not rsi_high and not rsi_low:
            step_reward -= 0.01 * (last_close_position_ticks - 1)

        return step_reward

    def _process_data(self):
        self.prices_close = self.df.loc[:, 'Close'].to_numpy()
        self.prices_open = self.df.loc[:, 'Open'].to_numpy()

        self.dates = self.df.index
        self.volumes = self.df.loc[:, 'Volume'].to_numpy()
        self.pivots = self.df[['Pivot_1', 'Pivot_2',
                               'Pivot_3', 'Pivot_4', 'Pivot_5']].to_numpy()
        self.rsi = self.df.loc[:, 'feature_rsi'].to_numpy()
        # pivots = self.df[['Pivot_1', 'Pivot_2','Pivot_3', 'Pivot_4', 'Pivot_5']].to_numpy()
        # Create a NumPy array containing the columns you want to process
        # selected_columns = ['Open', 'High', 'Low', 'Close','feature_vwap', 'feature_ema', 'feature_atr', 'feature_rsi']
        # selected_columns = ['High', 'Low', 'Close']
        # data_to_process = self.df[selected_columns].to_numpy()

        # Apply _calculate_candle_difference to the NumPy array
        # processed_data = self._calculate_table_difference(data_to_process)

        open = self.df["Open"] / self.df["Close"]
        high = self.df["High"] / self.df["Close"]
        low = self.df["Low"] / self.df["Close"]
        close = self.df["Close"].pct_change().values.reshape(-1, 1)

        input_features = np.zeros((len(self.dates), 2), dtype=np.float64).reshape(-1, 2)
        open = open.values.reshape(-1, 1)
        high = high.values.reshape(-1, 1)
        low = low.values.reshape(-1, 1)
        volume = self.df["Volume"].pct_change().values.reshape(-1, 1)
        rsi = self.df["feature_rsi"].pct_change().values.reshape(-1, 1)

        signal_features = np.hstack(
            (
                close,
                open,
                high,
                low,
                volume,
                rsi
                # input_features
            )
        )

        # signal_features[:, -1] = self._initial_balance
        """ signal_features[:, -1] = 3
        signal_features[:, -2] = Positions.Long.value
        signal_features[:-3, -1] = 0.0
        signal_features[:-3, -2] = Positions.NoPosition.value """

        # clean nan
        signal_features[~np.isnan(signal_features).any(axis=1)]

        # Create Scaler
        self.sc.fit(signal_features)

        return signal_features

    def _get_observation(self):
        info = self._get_step_info()

        old_value = self._get_history("balance", -2)
        new_value = self._get_history("balance", -1)

        percentage_change = ((new_value - old_value)
                             / abs(old_value)) if old_value != 0 else 0.0

        """ # Update Porfolio Pnl
        self.features[self._current_tick, -1] = percentage_change

        # Update current position
        self.features[self._current_tick, -2] = info['position'].value """

        # Get the next observation window
        observation = self.features[self._current_tick
                                    + 1 - self.windows: self._current_tick + 1].copy()

        # Apply transformation
        observation = self.sc.transform(observation)

        return observation

        # stacked_observations = []
        # for i in range(len(observation) - self.lstm_window):
        #    stacked_observations.append(observation[i:i + self.lstm_window])

        # return stacked_observations

    def _calculate_table_difference(self, data):
        # Use list comprehension to calculate differences for all columns
        candle_diff_array = np.array([np.diff(col_data, prepend=0)
                                     for col_data in data.T]).T

        # Set 0 on the 1st row for all columns
        candle_diff_array[0, :] = 0

        return candle_diff_array

    def _calculate_candle_difference(self, data):
        # Calculate the difference between the current candle and the last candle
        candle_diff = np.diff(data, axis=0)

        # Add a 0 at the beginning to match the shape
        candle_diff_array = np.insert(candle_diff, 0, 0, axis=0)

        return candle_diff_array

    def _get_step_info(self):
        return {
            "tick": self._current_tick,
            "total_reward": self._total_reward,
            "balance": self._porfolio.get_balance(),
            "total_profit": self._porfolio.get_total_pnl(),
            "open_pnl": self._porfolio.get_open_pnl(),
            "position": self._position,
            "action": self._action,
        }

    def _get_history(self, label, index=-1):
        # Return history
        return self._all_history[self._reset_counter][index][label]

    def render(self, mode='human'):
        # Create a new figure for Price

        fig1, ax1 = plt.subplots(figsize=(12, 3))

        # date_strings = [str(date) for date in self.dates]
        date_strings = [str(date) if (date.hour in [9, 12, 15, 18]) and date.minute == 0 else
                        str(i) for i, date in enumerate(self.dates)]

        ax1.plot(date_strings, self.prices_close)
        # Customize x-axis labels
        ax1.set_xticks(range(len(date_strings)))
        ax1.set_xticklabels(date_strings, rotation=90, fontsize=6)

        render_hist = self._all_history[self._reset_counter - 1]
        short_ticks = [i for i, obj in enumerate(
            render_hist) if obj['action'] == Actions.Sell]
        long_ticks = [i for i, obj in enumerate(
            render_hist) if obj['action'] == Actions.Buy]
        close_ticks = [i for i, obj in enumerate(
            render_hist) if obj['action'] == Actions.NoAction]

        plt.plot(short_ticks, self.prices_close[short_ticks], 'ro', label='Short')
        plt.plot(long_ticks, self.prices_close[long_ticks], 'go', label='Long')
        plt.plot(close_ticks, self.prices_close[close_ticks], 'bx', label='Close')

        short_position_ticks = [i for i, obj in enumerate(
            render_hist) if obj['position'] == Positions.Short]
        long_position_ticks = [i for i, obj in enumerate(
            render_hist) if obj['position'] == Positions.Long]
        no_position_ticks = [i for i, obj in enumerate(
            render_hist) if obj['position'] == Positions.NoPosition]

        max_price = max(self.prices_close)  # Get the maximum price from your data
        max_price += 10
        plt.plot(short_position_ticks, [
                 max_price] * len(short_position_ticks), 'rs', label='Short Position')
        plt.plot(long_position_ticks, [max_price]
                 * len(long_position_ticks), 'gs', label='Long Position')
        """ plt.plot(no_position_ticks, [max_price]
                 * len(no_position_ticks), 'bs', label='No Position') """

        max_price -= 5
        plt.plot(short_ticks, [
                 max_price] * len(short_ticks), 'ro', label='Short Action')
        plt.plot(long_ticks, [max_price]
                 * len(long_ticks), 'go', label='Long Action')
        """ plt.plot(close_ticks, [max_price]
                 * len(close_ticks), 'bo', label='No Position Action') """

        total_reward = render_hist[-1]['total_reward']
        total_profit = render_hist[-1]['total_profit']

        # Count actions
        action_counts = Counter(obj['action'] for obj in render_hist)

        total_action_buy = action_counts[Actions.Buy]
        total_action_sell = action_counts[Actions.Sell]
        total_action_close = action_counts[Actions.NoAction]

        plt.title("Total Reward: %.6f ~ Total Profit: %.6f" %
                  (total_reward, total_profit))
        plt.suptitle("Total Long: %.1f ~ Total Short: %.1f ~ Total Close: %.1f" % (
            total_action_buy, total_action_sell, total_action_close))

        # Add legend
        ax1.legend()
        plt.tight_layout()

        # Rewards
        total_rewards = [obj['total_reward'] for obj in render_hist]
        # Calculate the number of zeros needed to fill the array
        num_zeros = len(date_strings) - len(total_rewards)

        # Fill total_rewards with zeros
        total_rewards = np.concatenate((total_rewards, np.zeros(num_zeros)))

        fig3, ax3 = plt.subplots(figsize=(12, 2))
        ax3.plot(date_strings, total_rewards)
        ax3.set_title('Rewards')
        ax3.set_xticklabels(date_strings, rotation=0, fontsize=1)
        plt.tight_layout()

        # Pnl
        total_profits = [obj['total_profit'] for obj in render_hist]
        # Fill total_rewards with zeros
        total_profits = np.concatenate((total_profits, np.zeros(num_zeros)))
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(date_strings, total_profits)
        ax2.set_title('Pnl')
        ax2.set_xticklabels(date_strings, rotation=0, ha='right', fontsize=1)
        plt.tight_layout()

        # Create a new figure for Volume
        """ fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(self.dates, self.volumes)
        ax2.set_title('Volume')

        plt.tight_layout()
        plt.show() """

        plt.show()

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)
