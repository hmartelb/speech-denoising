import argparse
import os
import tarfile

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class UrbanSound8KDataset(Dataset):
    '''
    UrbanSound8KDataset (Pytorch Dataset)
    Class containing the data loader for the UrbanSound8K dataset.

    '''

    def __init__(self, file_path, folders, csv_name='UrbanSound8K.csv', target_sr=16000, length_seconds=4):
        # Check if the dataset exists, download dataset otherwise
        self.file_path = file_path
        self._try_download()

        # Load the file_names, labels and folders
        data = pd.read_csv(os.path.join(file_path, 'metadata', csv_name))
        self.file_names = [data['slice_file_name'].iloc[i]
                           for i in range(len(data)) if data['fold'].iloc[i] in folders]
        self.labels = [data['class'].iloc[i]
                       for i in range(len(data)) if data['fold'].iloc[i] in folders]
        self.folders = [data['fold'].iloc[i] for i in range(
            len(data)) if data['fold'].iloc[i] in folders]

        # Audio transformations
        self.target_sr = target_sr
        self.length_seconds = length_seconds

    def _try_download(self):
        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)
            filename = os.path.join(self.file_path, 'UrbanSound8K.tar.gz')
            zenodo_url = 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1'
            print(f'UrbanSound8K will be downloaded from {zenodo_url}')
            print(f'Destination path: {self.file_path}')

            print("\nDownload started:")
            r = requests.get(zenodo_url, stream=True)
            total = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as file, tqdm(
                    desc=filename,
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in r.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

            print(f"\nExtracting file: {filename}")
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall()

    @property
    def segment_length(self):
        return self.target_sr * self.length_seconds

    def __getitem__(self, index):
        filename = os.path.join(
            self.file_path, 'audio', f'fold{self.folders[index]}', self.file_names[index])

        # Load and (downmix to mono if needed)
        audio, original_sr = torchaudio.load(filename)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        # Resample from original_sr to target_sr (default: 16kHz)
        if original_sr != self.target_sr:
            audio = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)(audio)

        audio = audio.permute(1, 0)
        # Return a fixed-length segment
        pad = self.segment_length - audio.shape[0]
        if pad > 0:
            audio = torch.cat([audio, torch.zeros([pad, 1])], dim=0)
        if pad < 0:
            audio = audio[:self.segment_length]

        audio = audio.permute(1, 0)
        return audio, self.labels[index]

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--original_sr', default=44100, type=int)
    ap.add_argument('--target_sr', default=16000, type=int)
    ap.add_argument('--length_seconds', default=4, type=int)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    train_data = UrbanSound8KDataset(
        args.dataset_path,
        folders=range(1, 10),
        original_sr=args.original_sr,
        target_sr=args.target_sr,
        length_seconds=args.length_seconds
    )
    test_data = UrbanSound8KDataset(
        args.dataset_path,
        folders=[10],
        original_sr=args.original_sr,
        target_sr=args.target_sr,
        length_seconds=args.length_seconds
    )

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    '''
    Unit tests
    Plot and save the audios under the 'unit_tests' folder.

    * Single example
    * Batch (data loader)
    '''

    from utils import plot_waveforms, save_waveforms
    from tests import shape_test, waveforms_test, iteration_test

    UNIT_TESTS_DIR = os.path.join('unit_tests', 'UrbanSound8K_16kHz')
    if not os.path.isdir(UNIT_TESTS_DIR):
        os.makedirs(UNIT_TESTS_DIR)

    train_first_item, _ = train_data[0]
    print(train_first_item[0].shape)
    plot_waveforms(train_first_item, to_file=os.path.join(UNIT_TESTS_DIR, 'train_single.png'))
    save_waveforms(train_first_item, args.target_sr,
                   os.path.join(UNIT_TESTS_DIR, 'train_single'))

    assert list(train_first_item[0].shape) == [
        args.target_sr*args.length_seconds], "Wrong output shape in single example (train)"

    test_first_item, _ = test_data[0]
    print(test_first_item[0].shape)
    plot_waveforms(test_first_item, to_file=os.path.join(UNIT_TESTS_DIR, 'test_single.png'))
    save_waveforms(test_first_item, args.target_sr,
                   os.path.join(UNIT_TESTS_DIR, 'test_single'))

    assert list(test_first_item[0].shape) == [
        args.target_sr*args.length_seconds], "Wrong output shape in single example (test)"

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    for batch in train_loader:
        train_first_batch, _ = batch
        break
    print(train_first_batch.shape)
    plot_waveforms([train_first_batch[i, :]
                   for i in range(train_first_batch.shape[0])], to_file=os.path.join(UNIT_TESTS_DIR, 'train_batch.png'))
    save_waveforms([train_first_batch[i, :] for i in range(
        train_first_batch.shape[0])], args.target_sr, os.path.join(UNIT_TESTS_DIR, 'train_batch'))
    assert list(train_first_batch.shape) == [
        args.batch_size, 1, args.target_sr*args.length_seconds], "Wrong output shape in batch (train)"

    for batch in test_loader:
        test_first_batch, _ = batch
        break
    print(test_first_batch.shape)
    plot_waveforms([test_first_batch[i, :]
                   for i in range(test_first_batch.shape[0])], to_file=os.path.join(UNIT_TESTS_DIR, 'test_batch.png'))
    save_waveforms([test_first_batch[i, :] for i in range(
        test_first_batch.shape[0])], args.target_sr, os.path.join(UNIT_TESTS_DIR, 'test_batch'))
    assert list(test_first_batch.shape) == [
        args.batch_size, 1, args.target_sr*args.length_seconds], "Wrong output shape in batch (test)"

    iteration_test(train_loader)