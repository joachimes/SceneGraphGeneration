from torch_geometric.nn import GCNConv, GATConv
from torch.nn.functional import gelu, log_softmax
from torch.nn import Linear, CrossEntropyLoss, MSELoss 
from torch import stack
from torch.optim import Adam
from pytorch_lightning import LightningModule


class BaseGNN(LightningModule):
    def __init__(self,B_s,Node_Dim,Hidden_Dim,Out_Dim,Class_Dim,Loss_Only=True):
        super(BaseGNN,self).__init__()
        self.loss_only = Loss_Only
        self.Bs = B_s
        self.Conv1 = GCNConv(Node_Dim,Hidden_Dim)
        self.Conv2 = GCNConv(Hidden_Dim,Out_Dim)
        self.ClassHead = Linear(Out_Dim,Class_Dim)  
    

    def forward(self,Data):
        '''Simple SAGE Pass'''
        X, Edge_Index = Data.x, Data.edge_index
        X = gelu(self.Conv1(X,Edge_Index))
        X  = gelu(self.Conv2(X,Edge_Index))
        X = log_softmax(self.ClassHead(X),dim=1)
        return X
    

    def configure_optimizers(self):
        return Adam(self.parameters(),lr=1e-3, weight_decay=5e-4)
    

    def Loss(self,pred,Y):
        return CrossEntropyLoss()(pred, Y), MSELoss()(pred, Y)
    

    def training_step(self,batch,batch_idx):
        data = batch
        logits = self.forward(data)
        loss = self.Loss(logits, data.y)
        train_loss = {'train_loss':loss}
        # Acc_Bool = logits == data.y
        # Acc = sum(Acc_Bool.long()) * 100// len(logits)
        Result = {"loss":train_loss}
        return Result

    def training_epoch_end(self,Outputs):
        Avg_Loss = stack([x['loss'] for x in Outputs]).mean()
        Avg_Acc = stack([x['training_accuracy'] for x in Outputs]).mean()
        Epoch_Log = {"avg_training_loss":Avg_Loss,"avg_training_accuracy":Avg_Acc}
        self.logger.experiment.log_metrics(Epoch_Log)
        return Epoch_Log


    def validation_step(self, batch, batch_idx):
        val_data = batch
        logits = self.forward(val_data)
        loss = self.Loss(logits,val_data.y)
        acc_bool = logits == val_data.y
        acc = sum(acc_bool.long()) * 100// len(logits)
        res = {"val_loss":loss,"val_accuracy":acc.float()}
        self.logger.experiment.log_metrics(res)
        return res

    def test_step(self, batch, batch_idx):
        test_data = batch
        logits = self.forward(test_data)
        loss = self.Loss(logits,test_data.y)
        acc_bool = logits == test_data.y
        acc = sum(acc_bool.long()) * 100// len(logits)
        res = {"val_loss":loss,"val_accuracy":acc.float()}
        self.logger.experiment.log_metrics(res)
        return res

if __name__ == "__main__":
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers.csv_logs import CSVLogger

    seed_everything(42)
    Logger = CSVLogger("logs",name="Trial",version="SAGEConv")
    Logger.save()
    Mod = BaseGNN(2,50,150,200,121)
    trainer = Trainer(logger=Logger,max_epochs=1)
    trainer.fit(Mod)
    