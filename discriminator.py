import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

    def __init__(self, hidden_dim, output_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.gru = nn.GRU(output_size, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len x output_size
        emb = input.permute(1, 0, 2)                                 # seq_len x batch_size x output_size
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

if __name__ == "__main__":
    disc = Discriminator(hidden_dim=32, output_size=4, max_seq_len=10)

    seq_len = 10
    batch_size = 10

    inp = torch.randn((batch_size, seq_len, 4))
    target = torch.rand((batch_size, 1))

    out = disc.batchClassify(inp)
    print(out)

    loss = disc.batchBCELoss(inp, target)
    print(loss)

