import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=4, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, features):
        features = self.embedding(features)
        features = features.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        output = self.transformer_encoder(features)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2):
        super().__init__()
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead=4, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, memory):
        tgt = torch.zeros(seq_len, memory.size(1), memory.size(2)).cuda()
        output = self.transformer_decoder(tgt, memory)
        output = self.fc(output)
        return output


class TransformerScorer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size * 2)  
        encoder_layers = nn.TransformerEncoderLayer(hidden_size * 2, nhead=4, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        features = self.embedding(features)
        #features = features.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size * 2]
        output = self.transformer_encoder(features)
        output = self.fc(output)  # Apply linear layer without selecting the last position
        scores = self.sigmoid(output.squeeze(-1)) # Remove the last dimension (batch_size)
        return scores


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        
        self.e_transformer = TransformerEncoder(input_size, hidden_size, num_layers)
        self.d_transformer = TransformerDecoder(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, weighted_features):
        memory = self.e_transformer(weighted_features)
        h_mu, h_log_variance = memory.mean(dim=1), memory.std(dim=1)
        h = self.reparameterize(h_mu, h_log_variance)
        decoded_features = self.d_transformer(weighted_features.size(0), h.unsqueeze(0))
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_transformer = TransformerScorer(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features):
        scores = self.s_transformer(image_features)
        weighted_features = image_features * scores.view(-1, 1, 1)
        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)
        return scores, h_mu, h_log_variance, decoded_features