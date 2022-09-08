from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from graph_loader import Load_Graph
from models.base_model import GNN
import argparse
from torch.cuda import is_available


def main(args):
    seed_everything(42)
    
    dm = Load_Graph(args.data_dir)
    
    # logger = CSVLogger('logs', name='GNN')
    logger = CSVLogger("logs", name="GNN", version=str(args.batch_size)+"_"+str(args.max_epochs))
    # model = GNN(args.batch_size, args.node_dim, args.hidden_dim, args.out_dim, args.class_dim, args.loss_only)
    model = GNN(args.batch_size, 50, 500, 200, 121)
    trainer = Trainer(gpus=1 if is_available() else 0, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule=dm)
    trainer.test(model)


def test():
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import torch
    import pickle

    import json

    def load_obj(name):
        with open(name + '.json', 'rb') as f:
            return json.load(f)


    # Toy data
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    graph1 = Data(x=x, edge_index=edge_index.t().contiguous())
    graph2 = Data(x=x, edge_index=edge_index.t().contiguous())

    print(graph1)
    print(graph2)
    data_list = [graph1, graph2]

    loader = DataLoader(data_list, batch_size=32)
    print(next(iter(loader)))

    with open('train0.pkl', 'wb') as f:
        pickle.dump(data_list, f)

    #Own data:
    # train_graphs1 = load_obj('train0')
    train_graphs1 = load_obj('sgn-data-train/office_data')

    graph1 = train_graphs1[0]
    graph2 = train_graphs1[1]

    print(graph1)
    print(graph2)

    data_list = [graph1, graph2]

    loader = DataLoader(data_list, batch_size=32)
    print(next(iter(loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="sgn-data-train")

    Args = parser.parse_args()
    
    main(Args)
    # test()