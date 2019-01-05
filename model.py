import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, rnn_type, emb_size, hidden_size, num_layers, vocab_size, pad_idx):
        super(Network, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        if rnn_type == "gru":
            self.net = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional = True)
        elif rnn_type == "rnn":
            self.net = nn.RNN(emb_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        else:
            raise Exception("'{}' is an invalid network type".format(rnn_type))
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        batch_size, seq_len = x.size()

        embed = self.embedding(x)
        net_out, hidden = self.net(embed, hidden)
        
        # sum hidden activations of RNN across time
        net_out = net_out.view(batch_size, seq_len, 2, self.hidden_size)
        net_out = torch.sum(net_out, dim = 2)
        
        # take the average of the sequence in the feature space
        net_out = torch.sum(net_out, dim = 1) / seq_len

        logits = self.linear(net_out).squeeze()
        return logits, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)