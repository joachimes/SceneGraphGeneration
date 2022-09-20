import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningModule
from models.base_model import BaseGNN
from models.gat_model import GAT
from models.gcn_model import GCN
from models.gcngat_model import GCNGAT


class LightningTrainer(LightningModule):
    def __init__(self, model, node_dim, hidden_dim, out_dim, lr=1e-3, weight_decay=None, k=5, **kwargs):
        super(LightningTrainer,self).__init__()

        assert model in [BaseGNN.__name__, GAT.__name__, GCN.__name__, GCNGAT.__name__], 'Model not supported'
        self.model = eval(model)(node_dim, hidden_dim, out_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.k = k
        self.loss_fn = CrossEntropyLoss()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    

    def Loss(self, pred, Y):
        return self.loss_fn(pred, Y)
    
    
    def training_step(self, batch, batch_idx):
        data = batch
        out = self.model.forward(data)
        loss = self.Loss(out, data.y)
        Result = {'loss':loss}
        return Result

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss}
        self.logger.experiment.log_metrics(res)

    def validation_step(self, batch, batch_idx):
        val_data = batch
        out = self.model.forward(val_data)
        loss = self.Loss(out, val_data.y)
        pred = out.argmax(dim=1)
        val_correct = pred == val_data.y.argmax(dim=1)
        val_acc = int(val_correct.sum()) / len(val_correct)
        res = {'val_loss':loss, 'val_acc':val_acc}
        # self.logger.experiment.log_metrics(res)
        return res

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.FloatTensor([x['val_acc'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss, 'val_avg_acc': avg_acc}
        self.logger.experiment.log_metrics(res)
        self.log('val_loss', avg_loss)
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
        return res

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        for i in range(self.k):
            res[f'top_{i+1}_avg_test_acc'] = torch.FloatTensor([x[f'top_{i+1}_test_acc'] for x in outputs]).mean()
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
    