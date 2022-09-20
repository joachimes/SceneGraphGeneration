import torch
from torch.nn.functional import log_softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear, Parameter


class GCN(MessagePassing):
    def __init__(self, node_dim, hidden_dim, out_dim, **kwargs):
        super().__init__(aggr='mean')
        self.lin = Linear(node_dim, out_dim, bias=False)
        self.bias = Parameter(torch.Tensor(out_dim))

        self.reset_parameters()


    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        out += self.bias

        return log_softmax(out, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    