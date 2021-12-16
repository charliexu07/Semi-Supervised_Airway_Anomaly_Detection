import numpy as np
import torch
import math


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


def split(tr_x, tr_y, ratio):
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []

    every_nth = math.floor(1 / ratio)

    for i in range(len(tr_x)):
        if i % every_nth == 0:
            valid_x.append(tr_x[i])
            valid_y.append(tr_y[i])
        else:
            train_x.append(tr_x[i])
            train_y.append(tr_y[i])

    return np.asarray(train_x), np.asarray(train_y), np.asarray(valid_x), np.asarray(valid_y)


x = np.load('../data/tr_x.npy')
y = np.load('../data/tr_y.npy')

x, xmax, xmean = input_preprocess(x)
tr_x = x
tr_y = y.astype(int)

tr_x = torch.from_numpy(tr_x).cuda()
tr_y = torch.from_numpy(tr_y).cuda()

# comment this line if don't want to upsample
tr_x, tr_y = upsample(tr_x, tr_y)

ratio = 0.25
train_x, train_y, valid_x, valid_y = split(tr_x.tolist(), tr_y.tolist(), ratio)

print(tr_x.shape)
print(tr_y.shape)

print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)

np.save("../data/train_x.npy", train_x)
np.save("../data/train_y.npy", train_y)
np.save("../data/valid_x.npy", valid_x)
np.save("../data/valid_y.npy", valid_y)
