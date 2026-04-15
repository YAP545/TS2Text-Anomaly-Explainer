import torch
import torch.nn as nn

class AdvancedLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        # Optimized for multi-sensor inputs (Temp, Pressure, Vibration)
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hidden, _) = self.encoder(x)
        # Reconstruct the sequence
        x_decoded = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)
        x_decoded, _ = self.decoder(x_decoded)
        return self.output_layer(x_decoded)
