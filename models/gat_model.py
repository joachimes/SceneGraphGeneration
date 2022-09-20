import torch
from torch_geometric.nn import GATv2Conv
from torch.nn.functional import gelu, softmax, log_softmax
from torch.nn import Linear, Dropout, BatchNorm1d
from torch.optim import Adam


class GAT(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, heads=8, dropout=0.6, **kwargs):
        super(GAT,self).__init__()
        self.Conv1 = GATv2Conv(node_dim, hidden_dim, heads=heads)
        self.Bn1 = BatchNorm1d(hidden_dim*heads)

        self.Conv2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=1)
        self.Bn2 = BatchNorm1d(hidden_dim)

        self.ClassHead = Linear(hidden_dim, out_dim)
        self.Dropout = Dropout(p=dropout)


    def reset_parameters(self):
        self.Conv1.reset_parameters()
        self.Conv2.reset_parameters()
        self.ClassHead.reset_parameters()


    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.Dropout(gelu(self.Bn1(self.Conv1(x, edge_index))))
        x = self.Dropout(gelu(self.Bn2(self.Conv2(x, edge_index))))
        # x = softmax(self.ClassHead(x), dim=1)
        x = log_softmax(self.ClassHead(x), dim=1)
        # x = self.ClassHead(x)

        return x
    