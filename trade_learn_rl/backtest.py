from backtesting import Backtest, Strategy
from backtesting.test import SMA
from utils.load_data import load_file


def back_test(model, data):

    class SmaCross(Strategy):
        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, 100)
            self.ma2 = self.I(SMA, price, 200)
            self.last_action = 0
            self.model_data_buffer = []

        def prepare_model_data(self, current_data):
            model_data = [
                current_data.Close,
                current_data.Open,
                current_data.High,
                current_data.Low,
                current_data.feature_vwap,
                current_data.feature_ema,
                self.last_action,
                self.equity
            ]
            return model_data

        def next(self):
            if len(self.model_data_buffer) < 50:
                self.model_data_buffer.append(self.prepare_model_data(self.data))
            else:
                self.model_data_buffer.pop(0)
                self.model_data_buffer.append(self.prepare_model_data(self.data))
                print(self.model_data_buffer)
                action, _ = model.predict(self.model_data_buffer)
                self.last_action = 0

                if action > 1:
                    self.buy()
                    self.last_action = 1
                elif action < 1:
                    self.sell()
                    self.last_action = -1

    bt = Backtest(data, SmaCross, commission=.002, exclusive_orders=True)
    stats = bt.run()
    bt.plot()

# Call the back_test function with your model and data
# back_test(your_model, your_data)
