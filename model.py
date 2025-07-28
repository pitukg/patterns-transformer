import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_tokens):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,  # 4 heads for small d_model
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation='relu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)  # Binary classification

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.encoder(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Mean-pool over sequence (batch, d_model)
        logits = self.classifier(x)  # (batch, 2)
        return logits 