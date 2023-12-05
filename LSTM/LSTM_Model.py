import torch
from torch import nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        lstm_out, _ = self.lstm(embeds)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.linear(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs