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
        self.trade_pnl = self.calculate_pnl(current_price)
        return self.trade_pnl

    def close(self, close_price: float):
        self.is_trade_open = False
        self.trade_close_price = close_price
        self.trade_pnl = self.calculate_pnl(close_price)

    def calculate_pnl(self, current_price: float):
        pnl = 0.0
        if self.trade_direction == TradeDirection.Long:
            pnl = self.trade_amount * (current_price - self.trade_open_price)

        if self.trade_direction == TradeDirection.Short:
            pnl = self.trade_amount * (self.trade_open_price - current_price)

        return pnl

    def get_pnl(self):
        return self.trade_pnl

    def get_trade_cost(self):
        return self.trade_amount * self.trade_open_price

    def get_trade_profit(self):
        return self.trade_amount * self.trade_close_price


class Portofio():
    def __init__(self, balance: float):
        self.initial_balance = balance
        self.available_balance = self.initial_balance
        self.used_balance = 0
        self.trades = []
        self.current_trade = None
        self.open_pnl = 0.0
        self.porfolio_pnl = 0.0

    def open_trade(self, direction: TradeDirection, price: float, amount: float):
        trade_capital = price * amount
        if trade_capital > self.available_balance:
            raise Exception("Not enough balance")

        # Force - Close Last Trade
        if self.current_trade is not None:
            self.close_trade(price)

        self.openTrade = Trade(direction, price, amount)
        self.available_balance -= self.openTrade.get_trade_cost()
        self.used_balance = self.openTrade.get_trade_cost()

    def close_trade(self, price: float):
        if self.current_trade is None:
            raise Exception("No open trades")

        # Close Trade
        self.current_trade.close(price)

        # Update Balances
        self.available_balance += self.openTrade.get_trade_profit()
        self.used_balance = 0

        # Update Porfolio Pnl
        self.porfolio_pnl += self.openTrade.get_pnl()

        # Save trade
        self.trades.append(self.current_trade)
        self.current_trade = None

    def update(self, price: float):
        if self.current_trade is not None:
            self.open_pnl = self.current_trade.update(price)
        else:
            self.open_pnl = 0.0

    def get_balance(self):
        return self.available_balance + self.used_balance + self.open_pnl

    def get_pnl(self):
        return self.porfolio_pnl

    def get_open_pnl(self):
        return self.open_pnl
