import numpy
import argparse
import random

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from torch_geometric.data import Data

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from matplotlib import pyplot as plt
import time

import numpy as np

def input_preprocess(x, xmax=None, xmean=None):
    # x = x.transpose([0,2,1]).astype(np.float32)
    x = x.astype(np.float32)
    if xmax is None:
        xmax = x.max(axis=0)
        xmax[xmax == 0] = 1
    x = x / xmax
    if xmean is None:
        xmean = x.mean(axis=0)
    x = x - xmean
    return x, xmax, xmean

def upsample(tr_x, tr_y):
    new_tr_y, _ = torch.max(tr_y, dim=1)
    upsample_x = []
    upsample_y = []

    for index in range(new_tr_y.shape[0]):
        if new_tr_y[index].item() == 0:
            upsample_x.append(tr_x[index])
            upsample_y.append(tr_y[index])

    upsample_x = torch.stack(upsample_x * 70)
    upsample_y = torch.stack(upsample_y * 70)

    new_tr_x = torch.cat((tr_x, upsample_x))
    new_tr_y = torch.cat((tr_y, upsample_y))

    return new_tr_x, new_tr_y

def train(x):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(x, param['drop_feature_rate_1'])
    x_2 = drop_feature(x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:default.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    edge_index = torch.tensor(
        [[18, 18, 18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
          19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
         [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18,
          18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23]],
        dtype=torch.long).cuda()

    x = np.load('../data/train_x.npy').astype(np.float32)
    y = np.load('../data/train_y.npy').astype(int)

    tr_x = torch.from_numpy(x).cuda()
    tr_y = torch.from_numpy(y).cuda()

    data = Data(x=tr_x[0], edge_index=edge_index, y=tr_y[0])

    data = data.to(device)

    # generate split
    split = generate_split(24, train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    encoder = Encoder(10, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')

    losses = []
    start = time.time()

    # for epoch in range(1, param['num_epochs'] + 1):
    for epoch in range(1, param['num_epochs']):
        ite = 0
        for img in tr_x:
            single_loss = train(img)
            ite += 1
            if 'train' in log and ite % 50 == 0:
                print(f'(T) | Epoch={epoch:03d}, loss={single_loss:.4f}')

    end = time.time()
    print("Total time: " + str(end - start) + " seconds")

    PATH = "../model/train_model"
    torch.save(model.state_dict(), PATH)

    np.save("../model/train_loss.npy", np.array(losses))

    plt.plot(losses)
    plt.show()

    exit(0)
