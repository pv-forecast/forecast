import torch
import torch.nn as nn


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        # Hidden layer/dimensions
        self.hidden_dim = hidden_dim
        # Number of stacked LSTM's
        self.layer_dim = layer_dim # nummer erhöhen
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Dropout layer
        self.drop = nn.Dropout(p=0.5)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()


        # e.g. 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach())) # why detach?

        outd = self.drop(out)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        outd = self.fc(outd[:, -1, :])
        # out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        """print('The shape of lstm_out is:', out.shape)  # (batch_size, seq_len, hidden_dim)
        print('The shape of h is:', h_n.shape)"""
        return outd

    def loss(self, x, y):
        loss = torch.sqrt(self.criterion(x, y))
        return loss