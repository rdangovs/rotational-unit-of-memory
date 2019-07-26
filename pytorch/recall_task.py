import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from rum_model import RUM

import argparse
import os

parser = argparse.ArgumentParser(
    description='Recall Task')

parser.add_argument('--T', type=int, default=50,
                    help='length ')
parser.add_argument('--n_output', type=int, default=10,
                    help='number of output classes')
parser.add_argument('--n_iter', type=int, default=10000,
                    help='number of iterations')
parser.add_argument('--n_batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--n_hidden', type=int, default=50,
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


def recall_data(T, n_data):
    """ Creates the recall data. """

    # character
    n_category = int(T // 2)

    input1 = []
    for i in range(n_data):
        x0 = np.arange(1, n_category + 1)
        np.random.shuffle(x0)
        input1.append(x0[:T // 2])
    input1 = np.array(input1)
    # number
    input2 = np.random.randint(
        n_category + 1, high=n_category + 11, size=(n_data, T // 2))
    # question mark
    input3 = np.zeros((n_data, 2))
    seq = np.stack([input1, input2], axis=2)
    seq = np.reshape(seq, [n_data, T])
    # answer
    ind = np.random.randint(0, high=T // 2, size=(n_data))
    input4 = np.array([[input1[i][ind[i]]] for i in range(n_data)])

    x = np.concatenate((seq, input3, input4), axis=1).astype('int32')
    y = np.array([input2[i][ind[i]] for i in range(n_data)]) - n_category - 1

    return x, y


def next_batch(data_x, data_y, step, batch_size):
    data_size = data_x.shape[0]
    start = step * batch_size % data_size
    end = start + batch_size
    if end > data_size:
        end = end - data_size
        batch_x = np.concatenate((data_x[start:, ], data_x[:end, ]))
        batch_y = np.concatenate((data_y[start:], data_y[:end]))
    else:
        batch_x = data_x[start:end, ]
        batch_y = data_y[start:end]
    return batch_x, batch_y


def one_hot(data_x, data_y):
    input_data = torch.from_numpy(data_x).long().view(-1, 1)
    input_data = torch.zeros(
        input_data.size()[0], n_input).scatter_(1, input_data, 1)
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
            self.rnn = nn.GRU(n_input, hidden_size, batch_first=True)
        else:
            raise
        self.output = nn.Linear(hidden_size, n_output)
        self.hidden_size = hidden_size

    def forward(self, input):
        hidden_out, hidden_last = self.rnn(input)
        probabilities = F.log_softmax(self.output(
            hidden_last.view(-1, self.hidden_size)), dim=1)

        return probabilities


n_input = args.T // 2 + 10 + 1
n_train = 100000
n_valid = 1000
n_test = 2000
n_steps = args.T + 3


# create data
train_x, train_y = recall_data(args.T, n_train)
val_x, val_y = recall_data(args.T, n_valid)
test_x, test_y = recall_data(args.T, n_test)

val_input, val_labels = one_hot(val_x, val_y)
test_input, test_labels = one_hot(test_x, test_y)

# create model
net = Net(args.rnn_type, args.n_hidden, n_input,
          args.n_output, args.eta_, args.lambda_).cuda()

# loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)

# training
losses = []
val_losses = []
for step in range(args.n_iter):

    batch_x, batch_y = next_batch(train_x, train_y, step, args.n_batch)
    inputs, labels = one_hot(batch_x, batch_y)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print("Step %6d, Loss %.4f" % (step, losses[-1]))

    if step % 1000 == 0 and step > 0:
        print("Running validation")
        val_outputs = net(val_input)
        val_loss = criterion(val_outputs, val_labels)
        val_losses.append(val_loss.item())
        print("Validation loss %.4f" % (val_losses[-1]))

print("Running test")
test_outputs = net(test_input)
test_loss = criterion(test_outputs, test_labels)
test_losses = test_loss.item()
print("Test loss %.4f" % (test_losses))


torch.save({
    'step': args.n_iter,
    'model': net,
    'losses': losses,
    'val_losses': val_losses,
    'test_losses': test_losses,
}, os.path.join(args.save_dir, args.exp_name) + '.tar')
