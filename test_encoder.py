import sys

import ae_model
import common
import datasets

prop = common.yaml_load()

net = ae_model.load_model(prop['model_directory'] + 'pump.net')
train_dataset = datasets.TrainDataset('{MachineType}'.format(MachineType='pump'),
                                      train_dir=prop['dev_directory'],
                                      prop=prop['log_mel_spec'],
                                      do_transform=False)
x, y = train_dataset.__getitem__(1)
print(x)
for layer in net:
    print(layer(x).shape)
    sys.exit()
