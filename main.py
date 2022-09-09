from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from graph_loader import GraphLoader
from models.base_model import BaseGNN
import argparse
from torch.cuda import is_available


def main(args):
    seed_everything(42)
    
    dm = GraphLoader(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # logger = CSVLogger('logs', name='GNN')
    logger = CSVLogger("logs", name="GNN", version=str(args.batch_size)+"_"+str(args.max_epochs))
    # model = BaseGNN(args.batch_size, args.node_dim, args.hidden_dim, args.out_dim, args.class_dim, args.loss_only)
    model = BaseGNN(args.batch_size, 49, 100, 200, 49)
    trainer = Trainer(gpus=1 if is_available() else 0, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule=dm)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data/sgn-data-train")

    Args = parser.parse_args()
    
    main(Args)