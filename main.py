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
    batch = next(iter(dm.train_dataloader()))
    args.out_dim = len(batch.y[0])

    model_type = '_logSoft_'

    logger = CSVLogger(f'logs/log_{args.model}', name=dm.scene+model_type+str(args.max_epochs), version=str(args.batch_size)+'_'+str(args.max_epochs))
    model = LightningTrainer(**vars(args))
    checkpoint_callback = ModelCheckpoint(dirpath=f'logs/log_{args.model}/{dm.scene}{model_type}{args.max_epochs}', monitor='val_loss', mode='min', save_top_k=1)
    trainer = Trainer(gpus=1 if is_available() else 0, logger=logger, max_epochs=args.max_epochs, callbacks=[checkpoint_callback], enable_progress_bar=False)
    # trainer = Trainer(gpus=1 if is_available() else 0, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule=dm)
    model = LightningTrainer.load_from_checkpoint(checkpoint_callback.best_model_path, **vars(args))
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default='./data/sgn-data-train')
    parser.add_argument("--scene", type=str, default='bedroom')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    
    # Model parameters
    parser.add_argument("--model", type=str, default='GAT')
    parser.add_argument("--node_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=52)


    Args = parser.parse_args()
    for mod in ['GAT']:
        for scene in ['office']:
            Args.model = mod
            Args.scene = scene
            main(Args)
        # Args.model = mod
        # main(Args)
    # main(Args)