#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
from pathlib import Path
import random

from torch.utils.data import Dataset
import torch
import torchaudio
from numpy import load as np_load, ndarray

import numpy as np

from pympler import muppy, summary
import pandas as pd


__author__ = 'Konstantinos Drossos -- Tampere University, Nikita Kuzmin -- Lomonosov Moscow State University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):

    def __init__(self,
                 data_dir: Path,
                 split: str,
                 input_field_name: str,
                 output_field_name: str,
                 load_into_memory: bool,
                 settings_audio,
                 settings_features,
                 online_preprocessing=True,
                 indexes: np.array=None,
                 transforms=None) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Data directory with Clotho dataset files.
        :type data_dir: pathlib.Path
        :param split: The split to use (`development`, `validation`)
        :type split: str
        :param input_field_name: Field name for the input values
        :type input_field_name: str
        :param output_field_name: Field name for the output (target) values.
        :type output_field_name: str
        :param load_into_memory: Load the dataset into memory?
        :type load_into_memory: bool
        :param settings_audio: Settings about audio loading
        :type dict
        :param settings_features: Settings about audio processing
        :type dict
        :param indexes: Indexes of files, which depends on validation strategy
        :type indexes: numpy array
        :param transforms: List of transforms
        :type transforms: list
        """

        super(ClothoDataset, self).__init__()
        self.online_preprocessing = online_preprocessing
        the_dir: Path = data_dir.joinpath(split)
        self.split = split

        self.settings_audio = settings_audio
        self.settings_features = settings_features

        if indexes is None:
            self.examples: List[Path] = sorted(the_dir.iterdir())
        else:
            self.examples: List[Path] = list(np.array(sorted(the_dir.iterdir()))[indexes])
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory
        self.transforms = transforms
        self.resampler = torchaudio.transforms.Resample(orig_freq=settings_features['process']['sr'],
                                                        new_freq=settings_features['process']['sr_resample'])
        if load_into_memory:
            self.examples: List[ndarray] = [
                np_load(str(f), allow_pickle=True)
                for f in self.examples]
        self.cnt = 0

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray, Path]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values, and the Path of the file.
        :rtype: numpy.ndarray. numpy.ndarray, Path
        """

        ex = self.examples[item]
        if not self.load_into_memory:
            ex = np_load(str(ex), allow_pickle=True)
        if self.online_preprocessing:
            in_e = torchaudio.load(Path('data', 'clotho_audio_files', self.split, ex.file_name[0]))[0][0]
            ou_e = ex[self.output_name].item()
        else:
            in_e, ou_e = [ex[i].item()
                          for i in [self.input_name, self.output_name]]
        filename = ex.file_name[0]
        del ex
        if self.transforms is not None:
            for transform in self.transforms:
                in_e, ou_e = transform(dataset=self, inputs=(in_e, ou_e, filename))
        return in_e, ou_e, filename

    def random_sample(self, sample_audio=False):
        """
        Sampling audio or melspectrogram and encoded output
        :return:
        """

        item = random.randint(0, len(self.examples) - 1)
        ex = self.examples[item]
        if not self.load_into_memory:
            ex = np_load(str(ex), allow_pickle=True)
        if sample_audio:
            thedir = Path('./data/clotho_audio_files/').joinpath(self.split)
            filename = Path(thedir, ex.file_name[0])
            in_e = torchaudio.load(filepath=filename)[0][0]
            #in_e = self.resampler.forward(in_e)
            ou_e = ex[self.output_name].item()
        else:
            in_e, ou_e = [ex[i].item()
                          for i in [self.input_name, self.output_name]]

        return in_e, ou_e
# EOF
