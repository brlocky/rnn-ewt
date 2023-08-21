import math
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()

        # Calculate the number of hidden layers based on the provided hidden_size
        self.num_hidden_layers = max(
            0, int(math.log2(hidden_size)) - 4)  # Subtract 2

        # Dynamically create Linear layers and assign them to attributes
        for layer in range(self.num_hidden_layers + 1):
            in_features = hidden_size // (2 ** layer) if layer > 0 else input_size
            out_features = hidden_size // (2 ** (layer + 1)
                                           ) if layer < self.num_hidden_layers else output_size
            setattr(self, f'fc{layer+1}', nn.Linear(in_features, out_features))

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in range(1, self.num_hidden_layers + 1):
            x = self.relu(getattr(self, f'fc{layer}')(x))
        return getattr(self, f'fc{self.num_hidden_layers + 1}')(x)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        rnn_out, _ = self.rnn(input_seq)
        output = self.fc(rnn_out[:, -1, :])
        return output


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return lstm_out


class LSTMRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMRNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        output = self.fc(lstm_out[-1])  # Get the output from the last time step
        return output
