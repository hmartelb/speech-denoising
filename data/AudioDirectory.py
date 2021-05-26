import argparse
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDirectoryDataset(Dataset):
    '''
    Load all the audios recursively from a root directory (i.e. Dataset [train/test] split folder).
    '''
    def __init__(self, root, extensions=('.mp3', '.wav', '.flac'), keep_rate=1.0):
        self.filenames = list(self._find_files(root, extensions))
        if keep_rate > 0 and keep_rate < 1.0:
            self.filenames = np.random.choice(self.filenames, int(len(self.filenames)*keep_rate))
    
    def _find_files(self, root, extensions=('.mp3', '.wav', '.flac')):
        for base, dirs, files in os.walk(root):
            for f in files:
                filename = os.path.join(base, f)
                if filename.endswith(extensions):
                    yield filename

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        f = self.filenames[idx]
        audio, sr = torchaudio.load(f)
        return audio, sr

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    assert os.path.isdir(args.dataset_path), "dataset_path must be a valid directory"

    train_data = AudioDirectoryDataset(root=args.dataset_path)
    print("Number of examples:", len(train_data))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    from tests import iteration_test

    iteration_test(train_loader)
