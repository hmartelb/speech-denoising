# from .LibriSpeech import LibriSpeechDataset
# from .UrbanSound8K import UrbanSound8KDataset
from .NoiseMixer import NoiseMixerDataset
from .AudioDirectory import AudioDirectoryDataset

import os

LIBRISPEECH_PATH = os.path.join('..', 'datasets', 'LibriSpeech_16kHz_4s')
URBANSOUND8K_PATH = os.path.join('..', 'datasets', 'UrbanSound8K_16kHz_4s')

def get_dataset(name):
    if name == 'LibriSpeech':
        train_data = AudioDirectoryDataset(root=LIBRISPEECH_PATH)
        val_data = AudioDirectoryDataset(root=LIBRISPEECH_PATH)
        test_data = AudioDirectoryDataset(root=LIBRISPEECH_PATH)
        
    if name == 'UrbanSound8K':
        train_data = AudioDirectoryDataset(root=URBANSOUND8K_PATH)
        val_data = AudioDirectoryDataset(root=URBANSOUND8K_PATH)
        test_data = AudioDirectoryDataset(root=URBANSOUND8K_PATH)

    return train_data, val_data, test_data