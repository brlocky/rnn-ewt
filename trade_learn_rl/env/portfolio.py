from enum import Enum


class TradeDirection(Enum):
    Long = 0
    Short = 1


class Trade():
    def __init__(self, direction: TradeDirection, open_price: float, trade_amount: float):
        self.trade_direction = direction
        self.trade_open_price = open_price
        self.trade_close_price = open_price
        self.trade_amount = trade_amount
        self.is_trade_open = True
        self.trade_pnl = 0.0

    def update(self, current_price: float):
        self.trade_pnl = self.get_pnl(current_price)
        return self.trade_pnl

    def close(self, close_price: float):
        self.is_trade_open = False
        self.trade_close_price = close_price
        self.trade_pnl = self.get_pnl(close_price)
        return self.trade_pnl

    def get_closed_capital(self):
        return self.get_trade_initial_capital() + self.trade_pnl

    def _get_trade_fee(self):
        return self.trade_amount * self.trade_open_price * 0.001

    def get_trade_initial_capital(self):
        return self.trade_amount * self.trade_open_price

    def get_pnl(self, current_price: float):
        pnl = 0.0
        if self.trade_direction == TradeDirection.Long:
            pnl = self.trade_amount * (current_price - self.trade_open_price)

        if self.trade_direction == TradeDirection.Short:
            pnl = self.trade_amount * (self.trade_open_price - current_price)

        return pnl - self._get_trade_fee()


class Portfolio():
    def __init__(self, balance: float):
        self._initial_balance = balance
        self._available_balance = self._initial_balance
        self._trades = []
        self._current_trade = None
        self._open_pnl = 0.0
        self._porfolio_pnl = 0.0

    def open_trade(self, direction: TradeDirection, price: float, amount: float):
        trade_capital = price * amount
        if trade_capital > self._available_balance:
            raise Exception("Not enough balance")

        # Force - Close Last Trade
        if self._current_trade is not None:
            self.close_trade(price)

        self._current_trade = Trade(direction, price, amount)
        self._available_balance -= self._current_trade.get_trade_initial_capital()

    def close_trade(self, price: float):
        if self._current_trade is None:
            return

        # Close Trade
        trade_pnl = self._current_trade.close(price)

        # Update Balances
        self._available_balance += self._current_trade.get_closed_capital()

        # Update Porfolio Pnl
        self._porfolio_pnl += trade_pnl

        # Save trade
        self._trades.append(self._current_trade)
        self._current_trade = None

    def update(self, price: float):
        if self._current_trade is not None:
            self._open_pnl = self._current_trade.update(price)
        else:
            self._open_pnl = 0.0

    def get_balance(self):
        return self._available_balance + self._open_pnl

    def get_total_pnl(self):
        return self._porfolio_pnl + self._open_pnl

    def get_open_pnl(self):
        return self._open_pnl
