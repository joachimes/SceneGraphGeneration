import torch
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.utils import add_self_loops, degree
from torch.nn.functional import gelu, log_softmax
from torch.nn import Linear, Dropout, BatchNorm1d, Parameter



class GCNGAT(MessagePassing):
    def __init__(self, node_dim, hidden_dim, out_dim, heads=8, dropout=0.6, **kwargs):
        super().__init__(aggr='mean')

        self.Conv1 = GATv2Conv(node_dim, hidden_dim, heads=heads)
        self.Bn1 = BatchNorm1d(hidden_dim*heads)

        self.Conv2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=1)
        self.Bn2 = BatchNorm1d(hidden_dim)

        self.lin = Linear(hidden_dim, out_dim, bias=False)
        self.bias = Parameter(torch.Tensor(out_dim))
        self.Dropout = Dropout(p=dropout)

        self.reset_parameters()


    def reset_parameters(self):
        self.Conv1.reset_parameters()
        self.Conv2.reset_parameters()
        self.lin.reset_parameters()
        self.bias.data.zero_()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.Dropout(gelu(self.Bn1(self.Conv1(x, edge_index))))
        x = self.Dropout(gelu(self.Bn2(self.Conv2(x, edge_index))))
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
    