import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Generator(nn.Module):

    def __init__(self, hidden_dim, output_size, max_seq_len, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.output_size = output_size

        self.gru = nn.GRU(output_size, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, output_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size x output_size
        inp = inp.view(1, -1, self.output_size)               # 1 x batch_size x output_size
        out, hidden = self.gru(inp, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x output_size
        # out = F.log_softmax(out, dim=1)
        out = F.tanh(out)
        return out, hidden

    def sample(self, num_samples):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length x output_size (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len, self.output_size)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.zeros(num_samples, self.output_size))


        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x output_size
            samples[:, i] = out.data

            inp = out

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        loss_fn = nn.MSELoss()
        batch_size, seq_len, out_size = inp.size()
        inp = inp.permute(1, 0, 2)           # seq_len x batch_size
        target = target.permute(1, 0, 2)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len, _ = inp.size()
        inp = inp.permute(1, 0, 2)          # seq_len x batch_size
        target = target.permute(1, 0, 2)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -(torch.sum(target.data[i][j] - out[j])**2) * reward[j]
                #loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

if __name__ == "__main__":
    gen = Generator(hidden_dim=32, output_size=4, max_seq_len=10)
    samples = gen.sample(8)
    print(samples.size())

    seq_len = 10
    batch_size = 10

    inp = torch.randn((batch_size, seq_len, 4))
    target = torch.randn((batch_size, seq_len, 4))
    reward = torch.arange(batch_size)

    loss = gen.batchNLLLoss(inp, target)
    print(loss)

    loss = gen.batchPGLoss(inp, target, reward)
    print(loss)