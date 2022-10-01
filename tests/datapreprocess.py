import os
import sys

import numpy
import torch
import torchaudio
import librosa
import librosa.display
import audioread
from decorators import *
from matplotlib import pyplot as plt
import soundfile

path = r'D:\PycharmProjects\DCASE2020\data\dev_data\pump\test\anomaly_id_00_00000000.wav'
# path = r'D:\PycharmProjects\DCASE2020\data\sample.wav'
lis = [1,2,4]
print(lis[1:2])
y, sr = librosa.load(path, sr=None, mono=True)
noise = numpy.random.randn(y.size) * 0.0003
print(noise)
print(y)
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
# spec = numpy.abs(spec)
# plt.figure(figsize=(15, 12))
spec = librosa.power_to_db(spec)
librosa.display.specshow(spec, sr=sr, y_axis='mel', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
plt.show()

spec = torch.from_numpy(spec)
print(len(y))
print(spec.shape)
spec = spec.view(1, spec.shape[0], spec.shape[1])
spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)(spec)
spec = torchaudio.transforms.TimeMasking(time_mask_param=400)(spec)
spec = spec.detach().numpy()
librosa.display.specshow(spec[0], sr=sr, y_axis='mel', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
plt.show()
