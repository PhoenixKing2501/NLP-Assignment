import torch
import torch.nn as nn


# Normal Neural Network
class NN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_Model, self).__init__()
        self.hidden = nn.Linear(input_size,
                                hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size,
                                output_size)

    def forward(self, x):
        hidden = self.relu(self.hidden(x))
        return self.output(hidden)
# END NN_Model


# Recurrent Neural Network (Bidirectional)
class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,
                            output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # out shape: (batch_size, seq_length, hidden_size * 2)
        # average out the hidden states across the sequence
        out = out.mean(dim=1)
        # out shape: (batch_size, hidden_size * 2)
        return self.fc(out)
# END RNN_Model


# Long Short-Term Memory (Bidirectional)
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,
                            output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out shape: (batch_size, seq_length, hidden_size * 2)
        # average out the hidden states across the sequence
        out = out.mean(dim=1)
        # out shape: (batch_size, hidden_size * 2)
        return self.fc(out)
# END LSTM_Model


if __name__ == "__main__":

    # Create neural network models
    nn_model = NN_Model(5, 10, 2)
    rnn_model = RNN_Model(5, 10, 2)
    lstm_model = LSTM_Model(5, 10, 2)

    input_tensor = torch.randn(5, 5)
    nn_output = nn_model(input_tensor)

    input_tensor = torch.randn(5, 10, 5)
    rnn_output = rnn_model(input_tensor)
    lstm_output = lstm_model(input_tensor)

    print("Neural Network output shape:", nn_output.shape)
    print("RNN output shape:", rnn_output.shape)
    print("LSTM output shape:", lstm_output.shape)
