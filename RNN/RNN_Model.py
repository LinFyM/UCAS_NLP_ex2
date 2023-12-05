import torch
from torch import nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        rnn_out, _ = self.rnn(embeds)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.linear(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs