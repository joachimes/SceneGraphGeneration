import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningModule
from models.base_model import BaseGNN
from models.gat_model import GAT


class LightningTrainer(LightningModule):
    def __init__(self, model, node_dim, hidden_dim, out_dim, lr=1e-3, weight_decay=None, k=5, **kwargs):
        super(LightningTrainer,self).__init__()

        assert model in [BaseGNN.__name__, GAT.__name__], 'Model not supported'
        self.model = eval(model)(node_dim, hidden_dim, out_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.k = k


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    

    def Loss(self, pred, Y):
        return CrossEntropyLoss()(pred, Y)
    
    
    def training_step(self, batch, batch_idx):
        data = batch
        out = self.model.forward(data)
        loss = self.Loss(out, data.y)
        Result = {'loss':loss}
        return Result
        

    def validation_step(self, batch, batch_idx):
        val_data = batch
        out = self.model.forward(val_data)
        loss = self.Loss(out, val_data.y)
        pred = out.argmax(dim=1)
        val_correct = pred == val_data.y.argmax(dim=1)
        val_acc = int(val_correct.sum()) / len(val_correct)
        res = {'val_loss':loss, 'val_acc':val_acc}
        self.logger.experiment.log_metrics(res)
        self.log('val_loss', loss)
        return res

    def test_step(self, batch, batch_idx):
        test_data = batch
        out = self.model.forward(test_data)
        loss = self.Loss(out, test_data.y)
        res = {'test_loss': loss}

        # Compare topk accuracy for pred and test_data.y
        topK_pred = torch.topk(out, k=self.k, dim=1).indices
        max_y = test_data.y.argmax(dim=1)
        test_correct_accum = torch.zeros_like(max_y)
        for i in range(self.k):
            test_correct_accum += topK_pred[:,i].flatten() == max_y
            res[f'top_{i+1}_test_acc'] = int(test_correct_accum.sum()) / len(test_correct_accum)
        self.logger.experiment.log_metrics(res)
        return res


if __name__ == '__main__':
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers.csv_logs import CSVLogger
    from  graph_loader import GraphLoader


    seed_everything(42)
    Logger = CSVLogger('logs', name='Trial', version='GNNConv')
    Logger.save()

    args = {'model':'GAT', 'node_dim':6, 'hidden_dim':150, 'out_dim':43}
    Mod = LightningTrainer(**args)
    args_loader = {'data_dir':'data/sgn-data-train', 'batch_size':32, 'num_workers':4}
    train_loader = GraphLoader(**args_loader)
    trainer = Trainer(logger=Logger, max_epochs=1)
    trainer.fit(Mod, datamodule=train_loader)
    