from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

__author__ = 'Nikita Kuzmin -- Lomonosov Moscow State University'

class ProperHoldout:
    """
    """

    def __init__(self, train_size, settings_io):
        self.train_size = train_size
        self.data_dir = Path(
            settings_io['root_dirs']['data'],
            settings_io['dataset']['features_dirs']['output'])
        self.the_dir = self.data_dir.joinpath(settings_io['dataset']['features_dirs']['development'])
        self.examples: List[Path] = sorted(self.the_dir.iterdir())

    def split(self):
        self.examples = [str(example)[:-6] for example in self.examples]
        unique_audio_filenames = np.unique(self.examples)

        train_files, valid_files = train_test_split(unique_audio_filenames,
                                                    train_size=self.train_size, shuffle=True)
        train_idxs = [i for i, example in enumerate(self.examples) if example in train_files]
        valid_idxs = [i for i, example in enumerate(self.examples) if example in valid_files]

        return train_idxs, valid_idxs