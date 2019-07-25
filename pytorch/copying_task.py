import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from rum_model import RUM

import pdb
import argparse

parser = argparse.ArgumentParser(
    description='Copying Task')

parser.add_argument('--T', type=int, default=10,
                    help='delay')
parser.add_argument('--n_input', type=int, default=10,
                    help='number of input classes')
parser.add_argument('--n_output', type=int, default=9,
                    help='number of output classes')
parser.add_argument('--n_sequence', type=int, default=10,
                    help='length of sequence')
parser.add_argument('--n_iter', type=int, default=10000,
                    help='number of iterations')
parser.add_argument('--n_batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--n_hidden', type=int, default=100,
                    help='hidden size')
parser.add_argument('--eta_', type=float, default=None,
                    help='eta for time normalization')
parser.add_argument('--lambda_', type=int, default=0,
                    help='lambda for associative memory')
parser.add_argument('--rnn_type', type=str, default='RUM',
                    help='type of RNN')

args = parser.parse_args()


def copying_data(T, n_data, n_sequence):
    """generating the data"""
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, T - 1))
    zeros2 = np.zeros((n_data, T))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')

    return x, y


class Net(nn.Module):
    def __init__(self, rnn_type, hidden_size,
                 n_input, n_output, eta_=None, lambda_=None):
        super(Net, self).__init__()
        if rnn_type == 'RUM':
            self.rnn = RUM(n_input, hidden_size, eta_=eta_, lambda_=lambda_)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(n_input, hidden_size)
        else:
            raise
        self.output = nn.Linear(hidden_size, n_output)

    def forward(self, input):
        hidden_out, hidden_last = self.rnn(input)
        probabilities = F.log_softmax(self.output(hidden_out), dim=2)

        return probabilities.view(-1, 9)  # expands across batch and time dims


n_test = args.n_batch
n_train = args.n_iter * args.n_batch

# create data
train_x, train_y = copying_data(args.T, n_train, args.n_sequence)
test_x, test_y = copying_data(args.T, n_test, args.n_sequence)

# one_hot
input_data = torch.from_numpy(train_x).long().view(-1, 1)
input_data = torch.zeros(
    input_data.size()[0], args.n_input).scatter_(1, input_data, 1)
input_data = input_data.view(*train_x.shape, -1).cuda()

target_data = torch.from_numpy(train_y).long().cuda()

# create model
net = Net(args.rnn_type, args.n_hidden, args.n_input,
          args.n_output, args.eta_, args.lambda_).cuda()

# loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# training
for i in range(args.n_iter):
    inputs = input_data[args.n_batch*i:args.n_batch*(i+1), :, :]
    labels = target_data[args.n_batch*i:args.n_batch*(i+1), :]

    labels = labels.view(-1)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(loss.item())
