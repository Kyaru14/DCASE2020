import torch
from torch import nn


def get_model():
    model = nn.Sequential(
        # input
        nn.Linear(640, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        # encode
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 8),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        # decode
        nn.Linear(8, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        # output
        nn.Linear(128, 640)
    )
    return model


state_dict = torch.load(r'D:\PycharmProjects\DCASE2020\data\net\pump.net')
net = get_model()
net.load_state_dict(state_dict)
print(state_dict)
for param in net.parameters():
    print(param.shape)
