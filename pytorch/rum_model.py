import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def rotation_components(x, y, eps=1e-12):
    size_batch = x.size()[0]
    hidden_size = x.size()[1]

    # construct the 2x2 rotation
    u = F.normalize(x, p=2, dim=1, eps=eps)
    costh = torch.sum(u * F.normalize(y, p=2, dim=1, eps=eps),
                      dim=1).view(size_batch, 1)
    sinth = torch.sqrt(1 - costh ** 2).view(size_batch, 1)
    Rth = torch.cat((costh, -sinth, sinth, costh),
                    dim=1).view(size_batch, 2, 2)

    # get v and concatenate u and v
    v = F.normalize(y - torch.sum(u * y, 1).view(size_batch, 1)
                    * u, p=2, dim=1, eps=eps)
    tmp = torch.cat((u.view(size_batch, 1, hidden_size),
                     v.view(size_batch, 1, hidden_size)), dim=1)

    return (u.view(size_batch, hidden_size, 1),
            v.view(size_batch, hidden_size, 1), tmp, Rth)


def rotation_operator(x, y, eps=1e-12):
    hidden_size = x.size()[1]
    tmp_u, tmp_v, tmp, Rth = rotation_components(x, y, eps=eps)

    return (torch.eye(hidden_size, device=x.device) -
            torch.matmul(tmp_u, torch.transpose(tmp_u, dim0=1, dim1=2)) -
            torch.matmul(tmp_v, torch.transpose(tmp_v, dim0=1, dim1=2)) +
            torch.matmul(
                torch.matmul(torch.transpose(tmp, dim0=1, dim1=2), Rth), tmp))


def rotate(v1, v2, v):
    size_batch = v1.size()[0]
    hidden_size = v1.size()[1]

    U = rotation_components(v1, v2)

    h = v.view(size_batch, hidden_size, 1)
    return (v + (- torch.matmul(U[0], torch.matmul(U[0].transpose(1, 2), h))
                 - torch.matmul(U[1], torch.matmul(U[1].transpose(1, 2), h))
                 + torch.matmul(U[2].transpose(1, 2),
                                torch.matmul(U[3], torch.matmul(U[2], h)))
                 ).view(size_batch, hidden_size))


class RUMCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 eta_,
                 lambda_,
                 bias,
                 eps,
                 activation):
        super(RUMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eta_ = eta_
        self.lambda_ = lambda_

        self.bias = bias
        self.eps = eps
        if activation == None:
            activation = 'relu'
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu

        self.r_kernel = nn.Parameter(torch.Tensor(
            input_size + hidden_size, hidden_size))
        nn.init.orthogonal_(self.r_kernel)
        self.r_bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.constant_(self.r_bias, 1.0)
        self.u_kernel = nn.Parameter(torch.Tensor(
            input_size + hidden_size, hidden_size))
        nn.init.orthogonal_(self.u_kernel)
        self.u_bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.constant_(self.u_bias, 1.0)
        self.inp_emb_kernel = nn.Parameter(
            torch.Tensor(input_size, hidden_size))
        nn.init.xavier_uniform_(self.inp_emb_kernel)
        self.inp_emb_bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.constant_(self.inp_emb_bias, 0.0)

    def forward(self, inputs, hidden, assoc_mem):

        if hidden is None:
            hidden = torch.zeros(inputs.size(
                0), self.hidden_size, device=inputs.device)

        if assoc_mem is None and self.lambda_ == 1:
            assoc_mem = torch.eye(
                self.hidden_size, device=inputs.device).unsqueeze(0)

        # linear mappings
        r = torch.matmul(torch.cat((inputs, hidden), 1),
                         self.r_kernel) + self.r_bias
        u = torch.matmul(torch.cat((inputs, hidden), 1),
                         self.u_kernel) + self.u_bias
        x_emb = torch.matmul(inputs, self.inp_emb_kernel) + self.inp_emb_bias

        # computes the gate
        u = u.sigmoid()

        # rotation and nonlinearity
        if self.lambda_ == 0:
            hidden_new = rotate(x_emb, r, hidden)
        elif self.lambda_ == 1:
            tmp_rotation = rotation_operator(x_emb, r)
            assoc_mem = torch.matmul(assoc_mem, tmp_rotation)
            hidden_new = torch.matmul(
                assoc_mem, hidden.unsqueeze(-1)).squeeze(-1)
        else:
            raise

        c = F.relu(hidden_new + x_emb)
        new_h = u * hidden + (1 - u) * c
        if self.eta_:
            new_h = F.normalize(new_h, p=2, dim=1, eps=self.eps) * self.eta_

        return new_h, assoc_mem


class RUM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 eta_=None,
                 lambda_=0,
                 bias=True,
                 eps=1e-12,
                 activation=None):
        super().__init__()
        self.rum_cell = RUMCell(input_size, hidden_size,
                                eta_, lambda_, bias, eps, activation)

    def forward(self, input_, hidden=None, assoc_mem=None):
        outputs = []
        for x in torch.unbind(input_, dim=1):
            hidden, assoc_mem = self.rum_cell(x, hidden, assoc_mem)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1), hidden
