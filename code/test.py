import numpy
import argparse
import random

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.utils import get_base_model, get_activation
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


class Net(torch.nn.Module):
    def __init__(self, D_in, D_out, conv_number):
        super(Net, self).__init__()
        self.conv1 = GCNConv(D_in, 16)
        self.conv2 = GCNConv(16, D_out)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(D_in, D_out)
        self.conv_number = conv_number

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.conv_number == 1:
            x = self.conv4(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            for i in range(self.conv_number - 2):
                x = self.conv3(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


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

    x = np.load('../data/tr_x.npy')
    ts_x = np.load('../data/ts_x.npy')
    ts_y = np.load('../data/ts_y.npy')

    x, xmax, xmean = input_preprocess(x)
    ts_x_origin = np.copy(ts_x)
    ts_x, _, _ = input_preprocess(ts_x, xmax, xmean)
    ts_x = torch.from_numpy(ts_x).cuda()
    ts_y = torch.from_numpy(ts_y).cuda()

    ts_y = ts_y.cpu().detach().numpy()

    encoder = Encoder(10, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)

    PATH = "../model/train_model"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    model2 = Net(128, 2, 3).to(device)
    PATH = "../model/finetune_model"
    model2.load_state_dict(torch.load(PATH))
    model2.eval()

    start = time.time()

    ts_pred = []
    for index2 in range(ts_x.shape[0]):
        data = Data(x=model(ts_x[index2], edge_index), edge_index=edge_index)
        pred = model2(data)
        one_ts_pred = torch.argmax(pred, dim=1)
        one_ts_pred, _ = torch.max(one_ts_pred, dim=0)
        ts_pred.append(one_ts_pred.item())

    ts_pred = torch.tensor(ts_pred).cpu().detach().numpy()

    end = time.time()

    print("Total time: " + str(end - start) + " seconds")

    test_acc = (ts_pred == ts_y).sum(dtype=ts_pred.dtype) / (len(ts_y))
    print('test acc', test_acc)

    test_recall = (ts_pred[ts_y == 1]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y == 1]))
    print('test recall', test_recall, (ts_pred[ts_y == 1]).sum(dtype=ts_pred.dtype), '/', len(ts_y[ts_y == 1]))

    test_sensitivity = (1 - ts_pred[ts_y == 0]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y == 0]))
    print('test sensitivity', test_sensitivity, (1 - ts_pred[ts_y == 0]).sum(dtype=ts_pred.dtype), '/',
          (len(ts_y[ts_y == 0])), "\n")
