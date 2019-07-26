import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from rum_model import RUM

import pdb
import argparse
import os

parser = argparse.ArgumentParser(
    description='Copying Task')

parser.add_argument('--T', type=int, default=500,
                    help='delay')
parser.add_argument('--n_input', type=int, default=10,
                    help='number of input classes')
parser.add_argument('--n_output', type=int, default=9,
                    help='number of output classes')
parser.add_argument('--n_sequence', type=int, default=10,
                    help='length of sequence')
parser.add_argument('--n_iter', type=int, default=1000,
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
parser.add_argument('--save_dir', type=str, default='./train_log',
                    help='save directory')
parser.add_argument('--exp_name', type=str, default='test',
                    help='name of experiment')

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

def one_hot(data_x, data_y):
    input_data = torch.from_numpy(data_x).long().view(-1, 1)
    input_data = torch.zeros(
        input_data.size()[0], args.n_input).scatter_(1, input_data, 1)
    input_data = input_data.view(*data_x.shape, -1).cuda()
    target_data = torch.from_numpy(data_y).long().cuda()
    return input_data, target_data

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

# create model
net = Net(args.rnn_type, args.n_hidden, args.n_input,
          args.n_output, args.eta_, args.lambda_).cuda()

# loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)


# training
losses = []
for i in range(args.n_iter):
    batch_x, batch_y = copying_data(args.T, args.n_batch, args.n_sequence)
    inputs, labels = one_hot(batch_x, batch_y)
    labels = labels.view(-1)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print("Step %6d, Loss %.4f" % (i, losses[-1]))

print("Running test")
test_x, test_y = copying_data(args.T, args.n_batch * 10, args.n_sequence)
inputs, labels = one_hot(test_x, test_y)
labels = labels.view(-1)
outputs = net(inputs)
loss = criterion(outputs, labels)
test_loss = loss.item()
print("Test loss %.4f" % (test_loss))

torch.save({
    'step': args.n_iter,
    'model': net,
    'losses': losses,
    'test_loss': test_loss,
}, os.path.join(args.save_dir, args.exp_name) + '.tar')
