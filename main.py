import argparse
from torch.cuda import is_available
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from graph_loader import GraphLoader
from train import LightningTrainer


def main(args):
    seed_everything(42)
    
    dm = GraphLoader(**vars(args))
    
    logger = CSVLogger('logs', name=args.model, version=str(args.batch_size)+'_'+str(args.max_epochs))
    model = LightningTrainer(**vars(args))
    checkpoint_callback = ModelCheckpoint(dirpath=f'logs/{args.model}', monitor='val_loss', mode='min', save_top_k=1)
    trainer = Trainer(gpus=1 if is_available() else 0, logger=logger, max_epochs=args.max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default='./data/sgn-data-train')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    
    # Model parameters
    parser.add_argument("--model", type=str, default='GAT')
    parser.add_argument("--node_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=150)
    parser.add_argument("--out_dim", type=int, default=43)


    Args = parser.parse_args()
    
    main(Args)