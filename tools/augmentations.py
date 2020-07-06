import numpy as np
import random
from tools.features_log_mel_bands import feature_extraction, from_mel_to_audio, from_mel_to_stft
from pathlib import Path
import pysndfx
import gc

import copy

from tools.file_io import load_audio_file
import torch


__author__ = 'Nikita Kuzmin -- Lomonosov Moscow State University'

class MixUp:

    def __init__(self, p, settings_features, simple_concat_captions=True,
                 sample_audio=False):

        self.p = p
        self.sample_audio = sample_audio
        self.settings_features = settings_features
        self.simple_concat_captions = simple_concat_captions

    def from_mel(self, mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def to_mel(self, hertz):
        return 2595.0 * np.log10(1 + hertz / 700.0)

    def mix_audio(self, first_audio, second_audio):

        a = np.random.uniform(0.4, 0.6)

        shorter, longer = first_audio, second_audio

        if shorter.shape[0] == longer.shape[0]:
            if self.sample_audio:
                return (longer + shorter) / 2.0
            else:
                longer = from_mel_to_audio(longer, **self.settings_features['process']) * a
                shorter = from_mel_to_audio(shorter,
                                            **self.settings_features['process'])
                return feature_extraction((longer + shorter) / 2, **self.settings_features['process'])

        if first_audio.shape[0] > second_audio.shape[0]:
            shorter, longer = longer, shorter


        if self.sample_audio:
            start = random.randint(0, longer.shape[0] - 1 - shorter.shape[0])
            end = start + shorter.shape[0]
            longer *= a
            longer[start:end] += shorter * (1 - a)
        else:
            longer = from_mel_to_audio(longer, **self.settings_features['process']) * a
            shorter = from_mel_to_audio(shorter,
                                        **self.settings_features['process'])
            start = random.randint(0, longer.shape[0] - 1 - shorter.shape[0])
            end = start + shorter.shape[0]
            longer[start:end] += shorter * (1 - a)
            longer = feature_extraction(longer,
                                        **self.settings_features['process'])

        return longer

    def mix_labels(self, first_labels, second_labels):
        if self.simple_concat_captions:
            return np.hstack([first_labels[:-1], second_labels[1:]])
        else:

            first_token = first_labels[0]
            last_token = first_labels[-1]
            first_labels = first_labels[1:-1]
            second_labels = second_labels[1:-1]
            res = np.empty((first_labels.size + second_labels.size,),
                           dtype=first_labels.dtype)
            min_size = min(first_labels.size, second_labels.size)
            res[0:2*min_size:2] = first_labels[:min_size]
            res[1:2*min_size:2] = second_labels[:min_size]
            if first_labels.size > second_labels.size:
                res[min_size * 2:] = first_labels[min_size:]
            elif second_labels.size > first_labels.size:
                res[min_size*2:] = second_labels[min_size:]
            res = np.concatenate(([first_token], res))
            res = np.concatenate((res, [last_token]))
            return res

    def mix_audio_and_labels(self,
                             first_audio, second_audio,
                             first_labels, second_labels):
        mixed_audio = self.mix_audio(first_audio, second_audio)
        mixed_labels = self.mix_labels(first_labels, second_labels)

        return mixed_audio, mixed_labels

    def __call__(self, dataset, inputs):
        resulted_audio, resulted_labels, filename = inputs[0], inputs[1], inputs[2]
        if np.random.uniform() <= self.p:
            random_sample = dataset.random_sample(sample_audio=self.sample_audio)
            resulted_audio, resulted_labels = self.mix_audio_and_labels(
                resulted_audio, random_sample[0],
                resulted_labels, random_sample[1]
            )
        return resulted_audio, resulted_labels


class AudioAugmentation:
    # https://github.com/ex4sperans/freesound-classification
    def __init__(self, p):

        self.p = p
        self.effects_chain = (
            pysndfx.AudioEffectsChain()
                .reverb(
                reverberance=random.randrange(50),
                room_scale=random.randrange(50),
                stereo_depth=random.randrange(50)
            )
                .pitch(shift=random.randrange(-300, 300))
                .overdrive(gain=random.randrange(2, 10))
                .speed(random.uniform(0.9, 1.1))
        )

    def __call__(self, dataset, inputs):

        resulted_audio = inputs[0]
        captions = inputs[1]
        del inputs
        gc.collect()
        if np.random.uniform() < self.p:
            resulted_audio = torch.from_numpy(self.effects_chain(resulted_audio.numpy()))
        return resulted_audio, captions

