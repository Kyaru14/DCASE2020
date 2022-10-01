import sys

from torch import nn
from torch.utils import data
from tqdm import tqdm

import ae_model
import common
import datasets
from decorators import *

prop = common.yaml_load()


@record_time
def train(models: dict,
          *,
          batch_size=8192,
          num_workers=0,
          num_epochs=30,
          lr=1e-3,
          start=0,
          end=6):
    gpu = torch.device('cuda')
    for mtype in prop['MachineType'][start:end]:
        with datasets.TrainDataset('{MachineType}'.format(MachineType=mtype),
                                   train_dir=prop['dev_directory'],
                                   prop=prop['log_mel_spec'],
                                   do_transform=True) as train_dataset:
            model = models[mtype]
            model.cuda()
            train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_func = nn.MSELoss()
            print(f'{mtype}:training start')
            for epoch in range(num_epochs):
                model.train()
                for x, y in tqdm(train_dataloader):
                    optimizer.zero_grad()
                    x = x.to(gpu).type(torch.float)
                    y = y.to(gpu).type(torch.float)
                    y_hat = model(x)
                    loss = loss_func(y_hat, y)
                    loss.backward()
                    optimizer.step()
                torch.save(model.state_dict(),os.path.join(prop['model_directory'], mtype + '.net'))
                print(f'epoch {epoch + 1}/{num_epochs}, loss:{loss}')
            print(f'{mtype}:training done')


if __name__ == '__main__':
    models = {}
    for mtype in prop['MachineType']:
        path = prop['model_directory'] + mtype + '.net'
        model = ae_model.get_model()
        if os.path.exists(path):
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
            print(mtype + ':state_dict found')
        else:
            print(mtype + ':state_dict not found')
        models[mtype] = model
    train(models, start=1, end=2)
