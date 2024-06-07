import torch
import torch.nn as nn

class TransformerDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(TransformerDiscriminator, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=4, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_h_last = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        features = self.embedding(features)
        features = features.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        output = self.transformer_encoder(features)
        h_last = self.fc_h_last(output[-1])  # Extract the last hidden state, [batch_size, hidden_size]
        prob = self.sigmoid(self.fc_out(h_last))  # [batch_size, 1]
        return h_last.unsqueeze(0), prob


