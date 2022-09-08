import glob
import json
import copy
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset, Data, LightningDataset


class graph_dataset(InMemoryDataset):
    scene = 'office'
    def __init__(self, root, transform=None, pre_transform=None):
        super(graph_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['']


    @property
    def processed_file_names(self):
        return self.scene + '_dataset.pt'


    def indices(self):
        return range(self.len())


    def process(self):
        with open(osp.join(self.raw_dir[:-4], self.scene + '_data.json')) as f:
            self.valid_rooms = json.load(f)
        

        id2type = np.loadtxt(osp.join('data','SUNCG_id2type.csv'), dtype=str, delimiter=',')
        dict_id2type = {}
        for line in id2type:
            dict_id2type[line[1]] = line[3]

        
        with open(osp.join('data', f'TRAIN_id2cat_{self.scene}.json'), 'r') as f:
            id2cat = json.load(f)


        cat2id = {id2cat[id]: id for id in id2cat.keys()}

        data_list = []
        for room in tqdm(self.valid_rooms):
            # loop for rooms
            node_list = self.__preprocess_root_wall_nodes__(room['node_list'])


            # Node features - {'co-occurrence':'un', 'support': 'dir', 'surround': 'dir'}
            attrs = []
            edges = []

            for _, node in node_list.items():
                # Grab edge connections 
                for neighbor in node['co-occurrence'] + node['support']: # + node['surround']:
                    to_node = node['self_info']['node_model_id'].split('_')[-1]
                    from_node = node_list[neighbor]['self_info']['node_model_id'].split('_')[-1]
                    edges.append([int(to_node), int(from_node)])



                # Grab node features
                if (node['type'] == 'root'):
                    cat = 'wall'
                    dim_vec = [0.0] * 3
                    pos_vec = [0.0] * 3
                elif (node['type'] == 'wall'):
                    cat = 'wall'
                    dim_vec = node['self_info']['dim']
                    pos_vec = node['self_info']['translation']
                else:
                    cat = dict_id2type[node['self_info']['node_model_id']]
                    dim_vec = node['self_info']['dim']
                    pos_vec = node['self_info']['translation']

                cat_vec = [0.0] * (len(cat2id.keys()) + 1)
                cat_vec[int(cat2id[cat])] = 1.0

                attrs.append(cat_vec + dim_vec + pos_vec)

            x = torch.tensor(attrs, dtype=torch.float)


            # Edge info
            # Normalize the edges indices
            # TODO: normalize model ids?
            edge_idx = torch.tensor(np.array(edges).transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
            map_edge = torch.zeros_like(edge_idx)
            for k,v in map_dict.items():
                map_edge[edge_idx==k] = v
            edge_index = torch.tensor(map_edge, dtype=torch.long)

            graph = Data(x=x, edge_index=edge_index)
            data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def to_torch(n, torch_type=torch.FloatTensor, requires_grad=False, dim_0=1):
        n = torch.tensor(n, requires_grad=requires_grad).type(torch_type)
        n = n.view(dim_0, -1)
        return n    
        
    ''' useful functions '''
 
    def __preprocess_root_wall_nodes__(self, node_list):
        """
        # simple preprocess for root and wall nodes
        :param node_list:
        :return:
        """

        node_list['root']['self_info'] = {'dim': [0.0, 0.0, 0.0], 'translation': [0.0, 0.0, 0.0],
                                          'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 'node_model_id': '0'}
        x_min = node_list['wall_0']['self_info']['translation'][0]
        x_max = node_list['wall_2']['self_info']['translation'][0]
        y_min = node_list['wall_3']['self_info']['translation'][2]
        y_max = node_list['wall_1']['self_info']['translation'][2]
        x_mean = 0.5 * (x_min + x_max)
        y_mean = 0.5 * (y_min + y_max)
        for wall_node in ['wall_0', 'wall_1', 'wall_2', 'wall_3']:
            node_list[wall_node]['self_info']['translation'][0] -= x_mean
            node_list[wall_node]['self_info']['translation'][2] -= y_mean

        # for root and wall nodes, switch cooc to support relation
        # (we simply assume all wall nodes are 'supported' by root node, all other nodes are 'supported' by wall nodes)
        if ('support' not in node_list['root'].keys()):
            node_list['root']['surround'] = []
            node_list['root']['support'] = copy.deepcopy(node_list['root']['co-occurrence'])
            node_list['root']['co-occurrence'] = []
        for cur_node in ['wall_0', 'wall_1', 'wall_2', 'wall_3']:
            if (len(node_list[cur_node]['co-occurrence']) > 0):
                node_list[cur_node]['surround'] = []
                node_list[cur_node]['support'] = copy.deepcopy(node_list[cur_node]['co-occurrence'])
                node_list[cur_node]['co-occurrence'] = []

        return node_list



class Load_Graph(LightningDataModule):
    def __init__(self, root):
        super(Load_Graph, self).__init__()
        self.graph_paths = glob.glob(osp.join(root, '*_data.json'))[0]
        self.num_workers = 1
        self.dataset = graph_dataset

        self.train_dataset = self.dataset(self.graph_paths, split='train')
        self.val_dataset = self.dataset(self.graph_paths, split='val')
        self.test_dataset = self.dataset(self.graph_paths, split='test')
        
        # self.train_dataset, self.val_dataset, self.test_dataset = self.dataset(self.graph_paths)


        self.dataloader = LightningDataset(train_dataset=self.train_dataset
                                            , val_dataset=self.val_dataset
                                            , test_dataset=self.test_dataset
                                            , num_workers=self.num_workers
                                            , pin_memory=True)

    def train_dataloader(self):
        return self.dataloader.train_dataloader()

    def val_dataloader(self):
        return self.dataloader.val_dataloader()

    def test_dataloader(self) :
        return self.dataloader.test_dataloader()


if __name__ == '__main__':
    root = './data/sgn-data-train/'
    graph = graph_dataset(root)
    