import torch
from torch_geometric.nn import GCNConv
from torch.nn.functional import relu, log_softmax
from torch.nn import Linear, Dropout, BatchNorm1d


class BaseGNN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, dropout=0.6, **kwargs):
        super(BaseGNN,self).__init__()
        self.Conv1 = GCNConv(node_dim, hidden_dim)
        self.Bn1 = BatchNorm1d(hidden_dim)

        self.Conv2 = GCNConv(hidden_dim, hidden_dim)
        self.Bn2 = BatchNorm1d(hidden_dim)

        self.ClassHead = Linear(hidden_dim, out_dim)
        self.Dropout = Dropout(p=dropout)


    def reset_parameters(self):
        self.Conv1.reset_parameters()
        self.Conv2.reset_parameters()
        self.ClassHead.reset_parameters()


    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.Dropout(relu(self.Bn1(self.Conv1(x, edge_index))))
        x  = self.Dropout(relu(self.Bn2(self.Conv2(x, edge_index))))
        x = log_softmax(self.ClassHead(x), dim=1)
        # x = self.ClassHead(x)
        return x
