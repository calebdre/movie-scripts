import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, vocab_size, pad_idx):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # vocab_size: vocabulary size
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx = pad_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # get embedding of characters
        embed = self.embedding(x)
        # pack padded sequence
#         embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, _ = self.rnn(embed)
        # undo packing
#         rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_size)
        rnn_out = torch.sum(rnn_out, dim=2)

        logits = self.linear(rnn_out)
        return logits

class GRU(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, vocab_size, pad_idx):
        super(GRU, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional = True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        batch_size, seq_len = x.size()

        embed = self.embedding(x)
        # fprop though RNN
        gru_out, hidden = self.gru(embed, hidden)
        
        # sum hidden activations of RNN across time
        gru_out = gru_out.view(batch_size, seq_len, 2, self.hidden_size)
        gru_out = torch.sum(gru_out, dim = 2)
        
        # take the average of the sequence in the feature space
        gru_out = torch.sum(gru_out, dim = 1) / seq_len

        logits = self.linear(gru_out).squeeze()
        return logits, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)