import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()

        # ----- Encoder -----
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]  # Last layer
        hidden = self.dropout(hidden)

        # Latent representation
        z = self.latent(hidden)

        # ----- Decoder -----
        decoder_hidden = self.decoder_input(z)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded_seq, _ = self.decoder(decoder_hidden)
        decoded_seq = self.output_layer(decoded_seq)

        return decoded_seq
