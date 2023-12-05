import torch
from torch import nn
import torch.nn.functional as F

class FNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = self.dropout(F.relu(self.linear1(embeds)))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs