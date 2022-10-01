import torch
from torch import nn


class Encoder(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            # input
            nn.Linear(640, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            # encode
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            # decode
            nn.Linear(10, 128),
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

    def forward(self, x):
        return self.net(x)


def get_model():
    model = nn.Sequential(
        Encoder(),
        Decoder()
    )
    return model


def load_model(filepath):
    model = get_model()
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    batch_size = 256
    x = torch.randn(size=(batch_size * 304, 640))
    for layer in get_model():
        x = layer(x)
        print(type(layer).__name__, x.shape)
