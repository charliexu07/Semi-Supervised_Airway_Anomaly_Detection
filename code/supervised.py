import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.data import Data
import time

edge_index = torch.tensor([[18, 18, 18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
                            19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18,
                            18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23]],
                          dtype=torch.long).cuda()


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


class Net2(torch.nn.Module):
    def __init__(self, D_in, D_out, conv_number):
        super(Net2, self).__init__()
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


x = np.load('../data/tr_x.npy')
y = np.load('../data/tr_y.npy')
ts_x = np.load('../data/ts_x.npy')
ts_y = np.load('../data/ts_y.npy')

x, xmax, xmean = input_preprocess(x)
ts_x_origin = np.copy(ts_x)
ts_x, _, _ = input_preprocess(ts_x, xmax, xmean)
tr_x = x
tr_y = y.astype(int)

tr_x = torch.from_numpy(tr_x).cuda()
tr_y = torch.from_numpy(tr_y).cuda()
ts_x = torch.from_numpy(ts_x).cuda()
ts_y = torch.from_numpy(ts_y).cuda()

tr_x, tr_y = upsample(tr_x, tr_y)
# print(tr_x.shape)

N, D_in, H, D_out = tr_x.shape[0], x.shape[1], 8, y.max() + 1
D_out = 2
D_in = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ts_y = ts_y.cpu().detach().numpy()

layer_num = [3]
l2_num = [1e-4]
lr_num = [1e-3]

# iter_num = 5
iter_num = 3
cnt = 0
val_acc_list = []

combo = 1

# 100% data

with open('result.txt', 'a') as f:
    f.write("Training using 100% of data for 500 epochs")
    f.write("\n")
    f.write("\n")

print("Training using 100% of data for 500 epochs")

for layers in layer_num:
    for l2 in l2_num:
        for lr in lr_num:
            cnt = cnt + 1
            val_acc_sum = 0
            val_acc_all = []
            print(
                'Train with {1} layers, {2} L2 regularization, and {3} learning rate:'.format(combo, layers,
                                                                                                         l2, lr), "\n")
            total_acc = []
            total_sensitivity = []
            total_specificity = []
            for ite in range(iter_num):
                start = time.time()
                model = Net2(D_in, D_out, layers).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
                model.train()
                # losses = []
                for epoch in range(500):
                    for index in range(tr_x.shape[0]):
                        optimizer.zero_grad()
                        data = Data(x=tr_x[index], edge_index=edge_index)
                        out = model(data)
                        loss = F.nll_loss(out, tr_y[index].long())
                        loss.backward()
                        # losses.append(loss)
                        optimizer.step()
                    print("epoch", epoch)

                # plt.plot(losses)
                # plt.show()

                model.eval()
                ts_pred = []
                for index2 in range(ts_x.shape[0]):
                    data = Data(x=ts_x[index2], edge_index=edge_index)
                    pred = model(data)
                    one_ts_pred = torch.argmax(pred, dim=1)
                    one_ts_pred, _ = torch.max(one_ts_pred, dim=0)
                    ts_pred.append(one_ts_pred.item())

                ts_pred = torch.tensor(ts_pred).cpu().detach().numpy()

                end = time.time()

                iter_msg = 'Iteration ' + str(ite + 1) + ": " + str(end - start) + " seconds"

                test_acc = (ts_pred == ts_y).sum(dtype=ts_pred.dtype) / (len(ts_y))
                total_acc.append(test_acc)
                print('test acc', test_acc)

                test_sensitivity = (ts_pred[ts_y == 1]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y == 1]))
                total_sensitivity.append(test_sensitivity)
                print('test sensitivity', test_sensitivity, (ts_pred[ts_y == 1]).sum(dtype=ts_pred.dtype), '/',
                      len(ts_y[ts_y == 1]))

                test_specificity = (1 - ts_pred[ts_y == 0]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y == 0]))
                total_specificity.append(test_specificity)
                print('test specificity', test_specificity, (1 - ts_pred[ts_y == 0]).sum(dtype=ts_pred.dtype), '/',
                      (len(ts_y[ts_y == 0])), "\n")

                with open('result.txt', 'a') as f:
                    f.write(iter_msg + "\n")
                    f.write('test acc: ' + str(test_acc) + "\n")
                    f.write('test sensitivity: ' + str(test_sensitivity) + "\n")
                    f.write('test specificity: ' + str(test_specificity) + "\n\n")

            print('average test acc', sum(total_acc) / len(total_acc))
            print('average test sensitivity', sum(total_sensitivity) / len(total_sensitivity))
            print('average test specificity', sum(total_specificity) / len(total_specificity), "\n")

            with open('result.txt', 'a') as f:
                f.write("final result:" + "\n")
                f.write('average test acc: ' + str(sum(total_acc) / len(total_acc)) + "\n")
                f.write('average test sensitivity: ' + str(sum(total_sensitivity) / len(total_sensitivity)) + "\n")
                f.write('average test specificity: ' + str(sum(total_specificity) / len(total_specificity)) + "\n\n\n")
