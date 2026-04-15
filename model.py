import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hidden, _) = self.encoder(x)
        x = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)
        x, _ = self.decoder(x)
        return self.output_layer(x)
