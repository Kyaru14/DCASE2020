import random
import sys

import librosa
import numpy
import torch
import torchaudio.transforms
from torch.utils import data
from tqdm import tqdm

from decorators import *


class Transform:
    @staticmethod
    def to_tensor(x: numpy.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x).type(torch.float)

    @staticmethod
    def pitch_shift(x: numpy.ndarray, sr: int, n_steps_range: tuple = (-1.0, 1.0)) -> numpy.ndarray:
        n_steps = n_steps_range[0] + (n_steps_range[1] - n_steps_range[0]) * random.random()
        return librosa.effects.pitch_shift(x, sr=sr, n_steps=n_steps)

    @staticmethod
    def time_stretch(x: numpy.ndarray, rate_range: tuple = (0.9, 1.2), **kwargs) -> numpy.ndarray:
        rate = rate_range[0] + (rate_range[1] - rate_range[0]) * random.random()
        return librosa.effects.time_stretch(x, rate=rate)

    @staticmethod
    def add_gaussian_noise(x: numpy.ndarray) -> numpy.ndarray:
        noise = numpy.random.randn(x.size) * numpy.average(x) * 0.05
        return x + noise

    @staticmethod
    def mixup(x1: numpy.ndarray, x2: numpy.ndarray, lambd=0.5):
        return lambd * x1 + (1 - lambd) * x2

    @staticmethod
    def masking(spec: numpy.ndarray):
        spec = torch.from_numpy(spec)
        spec = spec.view(1, spec.shape[0], spec.shape[1])
        spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)(spec)
        spec = torchaudio.transforms.TimeMasking(time_mask_param=160)(spec)
        return spec[0].detach().numpy()


class TrainDataset(data.Dataset):
    gpu = torch.device('cuda')

    def __init__(self,
                 machine_type: str,
                 *,
                 train_dir,
                 directory='train',
                 prop: dict,
                 do_transform: bool = True):
        # member fields
        self.machine_type: str = None  # single train set for a single machine type
        self.paths: list = []  # .wav file path
        self.prop: dict = prop  # yaml
        self.train_data_x: torch.Tensor = None  # preloaded data stored here
        self.train_data_y: torch.Tensor = None  # preloaded data stored here
        self.do_transform = do_transform
        # init
        self.machine_type = machine_type
        directory = os.path.join(train_dir, self.machine_type, directory)
        for wav_file in os.listdir(directory):
            self.paths.append(os.path.join(directory, wav_file))

    def load_from_path(self, path) -> tuple:
        x, sr = librosa.load(path, sr=None, mono=True)
        if self.do_transform and random.random() < 0.5:
            x2, _ = librosa.load(self.paths[random.randrange(0, len(self.paths))], sr=None, mono=True)
            x = Transform.mixup(x, x2)
        y = numpy.array(x)
        if self.do_transform and random.random() < 0.5:
            x = Transform.add_gaussian_noise(x)
        x = self.to_log_mel_spectrogram(x, sr, 0.3)
        y = self.to_log_mel_spectrogram(y, sr)
        x = Transform.to_tensor(x)
        y = Transform.to_tensor(y)
        return x, y

    def __getitem__(self, item) -> tuple:
        if (self.train_data_x is not None) and (self.train_data_y is not None):
            x, y = self.train_data_x[item], self.train_data_y[item]
        else:
            x, y = self.load_from_path(self.paths[item])
        return x, y

    def __len__(self):
        return len(self.train_data_x)

    def __enter__(self):
        self.train_data_x = torch.zeros((len(self.paths) * 309, self.prop['frames'] * self.prop['n_mels']))
        self.train_data_y = torch.zeros((len(self.paths) * 309, self.prop['frames'] * self.prop['n_mels']))
        for i, path in enumerate(tqdm(self.paths, desc='Preloading audios')):
            x, y = self.load_from_path(path)
            self.train_data_x[i * 309: (i + 1) * 309] = x
            self.train_data_y[i * 309: (i + 1) * 309] = y
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.train_data_x
        del self.train_data_y
        self.train_data_x = None
        self.train_data_y = None
        print('data unloaded')

    @property
    def mtype(self):
        return self.machine_type

    def to_log_mel_spectrogram(self, y, sr, p_masking: float = 0.0):
        """
        convert to mel_spectrogram
        :param p_masking: probability to do masking
        :param sr: sample rate
        :param y: audio : numpy.ndarray
        :return: mel_spec : numpy.ndarray
        """
        import librosa.display
        spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                              n_fft=self.prop['n_fft'],
                                              hop_length=self.prop['hop_length'],
                                              n_mels=self.prop['n_mels'])
        log_mel_spec = librosa.power_to_db(spec)
        if self.do_transform and random.random() < p_masking:
            log_mel_spec = Transform.masking(log_mel_spec)
        log_mel_spec = log_mel_spec.T
        frames = self.prop['frames']
        n_mels = self.prop['n_mels']
        output_length = len(log_mel_spec) - frames + 1
        y = numpy.zeros((output_length, n_mels * frames))
        for t in range(frames):
            y[:, n_mels * t: n_mels * (t + 1)] = log_mel_spec[t: t + output_length, :]
        return y
