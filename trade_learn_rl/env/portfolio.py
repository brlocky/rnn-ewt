from enum import Enum


class TradeDirection(Enum):
    Long = 0
    Short = 1


# fee_ratio = 0.005
fee_ratio = 0.01


class Trade():
    def __init__(self, direction: TradeDirection, open_price: float, trade_amount: float):
        self.trade_is_open = True
        self.trade_direction = direction
        self.trade_open_price = open_price
        self.trade_close_price = open_price
        self.trade_amount = trade_amount

    def increase_position(self, open_price: float, trade_amount: float):
        initial_cost = self.get_total_cost()
        # Calculate the weighted average open price
        new_open_price = ((self.trade_open_price * self.trade_amount)
                          + (open_price * trade_amount)) / (self.trade_amount + trade_amount)

        # Update the trade open price and trade amount
        self.trade_open_price = new_open_price
        self.trade_amount += trade_amount

        final_cost = self.get_total_cost()
        return final_cost - initial_cost

    def get_total_cost(self):
        return self.get_position_cost() + self.get_fee_cost()

    def get_position_cost(self):
        return self.trade_amount * self.trade_open_price

    def get_fee_cost(self):
        return self.get_position_cost() * fee_ratio

    def update(self, current_price: float):
        self.trade_close_price = current_price
        return self.get_pnl()

    def close(self, close_price: float):
        self.trade_is_open = False
        self.trade_close_price = close_price
        return self.get_pnl()

    def get_pnl(self):
        pnl = 0.0
        if self.trade_direction == TradeDirection.Long:
            pnl = self.trade_amount * (self.trade_close_price - self.trade_open_price)

        if self.trade_direction == TradeDirection.Short:
            pnl = self.trade_amount * (self.trade_open_price - self.trade_close_price)

        return pnl - self.get_fee_cost()


class Portfolio():
    def __init__(self, balance: float):
        self._initial_balance = balance
        self._available_balance = self._initial_balance
        self._trades = []
        self._current_trade = None
        self._porfolio_pnl = 0.0

    # Tick Update function
    def update(self, price: float):
        if self._current_trade is not None:
            self._current_trade.update(price)

        return self.get_open_pnl()

    # Open Trade or Increase Position
    def open_trade(self, direction: TradeDirection, price: float, amount: float) -> bool:
        trade_capital = price * amount

        # No capital available
        if trade_capital > self._available_balance:
            return False

        # Current Position has same direction - Increment position size
        """ if self._current_trade and self._current_trade.trade_direction == direction:
            new_position_cost = self._current_trade.increase_position(price, amount)
            self._available_balance -= new_position_cost
            return True """

        # Close position if open
        if self._current_trade:
            return False
            # self.close_trade(price)

        # Open new Position
        self._current_trade = Trade(direction, price, amount)
        self._available_balance -= self._current_trade.get_total_cost()

        return True

    # Close open trade
    def close_trade(self, price: float) -> float:
        if self._current_trade is None:
            return 0.0

        # Close Trade
        trade_pnl = self._current_trade.close(price)

        # Update Balances
        self._available_balance += self._current_trade.get_position_cost() + trade_pnl

        # Update Porfolio Pnl
        self._porfolio_pnl += trade_pnl

        # Save trade
        self._trades.append(self._current_trade)
        self._current_trade = None

        return trade_pnl

    # Account total Balance
    def get_balance(self):
        if self._current_trade:
            return self._available_balance + self._current_trade.get_position_cost() + self._current_trade.get_pnl()

        return self._available_balance

    # Account available Balance
    def get_available_balance(self):
        return self._available_balance

    # Realized and open PnL
    def get_total_pnl(self):
        return self.get_closed_pnl() + self.get_open_pnl()

    # Total Pnl Percentage relative to balance
    def get_total_pnl_percentage(self):
        total_pnl = self.get_total_pnl()
        available_balance = self.get_available_balance()

        # Calculate percentage
        if available_balance > 0:
            percentage = (total_pnl / available_balance) * 100
        else:
            # Handle the case where available_balance is 0 to avoid division by zero
            percentage = 0  # You can choose another appropriate value here

        return percentage

    # Realized PnL
    def get_closed_pnl(self):
        return self._porfolio_pnl

    # Current Position PnL
    def get_open_pnl(self):
        if self._current_trade is not None:
            return self._current_trade.get_pnl()
        return 0.0

    def get_open_pnl_percentage(self):
        if self._current_trade is not None:
            return self.get_trade_pnl_percentage(self._current_trade)

        return 0.0

    # Realized PnL
    def get_last_trade_pnl_percentage(self):
        # Find the last closed trade
        last_closed_trade = None
        for trade in reversed(self._trades):
            if trade.trade_is_open is False:
                last_closed_trade = trade
                break

        if last_closed_trade is None:
            return 0.0

        return self.get_trade_pnl_percentage(last_closed_trade)

    # PnL relative to balance
    def get_trade_pnl_percentage(self, trade):
        trade_pnl = trade.get_pnl()
        account_balance = self.get_balance()
        return (trade_pnl / account_balance) * 100
