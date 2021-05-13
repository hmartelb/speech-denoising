import argparse
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    '''
    LibriSpeechDataset (Pytorch Dataset)
    Class containing the data loader for the LibriSpeech dataset using the LibriSpeech class provided in torchaudio.

    '''

    def __init__(self, dataset_path, split, original_sr=16000, target_sr=16000, length_seconds=4):
        self.dataset_path = dataset_path
        self.split = split

        self.data = torchaudio.datasets.LIBRISPEECH(
            dataset_path, url=split, download=True)

        # Audio transformations
        self.original_sr = original_sr
        self.target_sr = target_sr
        self.length_seconds = length_seconds
        self.resampling_fn = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=target_sr)

    @property
    def segment_length(self):
        return self.target_sr * self.length_seconds

    def __getitem__(self, index):
        # self.data returns a tuple: (audio, sr, utterance, speaker_id, chapter_id, utterance_id)
        audio, _, utterance = self.data[index][0:3]
        # TODO: Maybe change this so that the audio is not permuted twice
        audio = audio.permute(1, 0)

        # Resample from 16kHz to target_sr (default: 16kHz)
        # TODO: Is this step necessary? LibriSpeech sr is 16000
        if self.original_sr != self.target_sr:
            audio = self.resampling_fn(audio)

        # Return a fixed-length segment
        pad = self.segment_length - audio.shape[0]
        if pad > 0:
            audio = torch.cat([audio, torch.zeros([pad, 1])], dim=0)
        if pad < 0:
            audio = audio[:self.segment_length]

        audio = audio.permute(1, 0)
        return audio, utterance

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--target_sr', default=16000, type=int)
    ap.add_argument('--length_seconds', default=4, type=int)
    ap.add_argument('--batch_size', default=4, type=int)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    train_data = LibriSpeechDataset(
        args.dataset_path,
        split="train-clean-100",
        target_sr=args.target_sr,
        length_seconds=args.length_seconds
    )
    test_data = LibriSpeechDataset(
        args.dataset_path,
        split="test-clean",
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

    UNIT_TESTS_DIR = os.path.join('unit_tests', 'LibriSpeech-100')
    if not os.path.isdir(UNIT_TESTS_DIR):
        os.makedirs(UNIT_TESTS_DIR)

    train_first_item, _ = train_data[0]
    print(train_first_item[0].shape)
    plot_waveforms(train_first_item, to_file=os.path.join(
        UNIT_TESTS_DIR, 'train_single.png'))
    save_waveforms(train_first_item, args.target_sr,
                   os.path.join(UNIT_TESTS_DIR, 'train_single'))

    assert list(train_first_item[0].shape) == [
        args.target_sr*args.length_seconds], "Wrong output shape in single example (train)"

    test_first_item, _ = test_data[0]
    print(test_first_item[0].shape)
    plot_waveforms(test_first_item, to_file=os.path.join(
        UNIT_TESTS_DIR, 'test_single.png'))
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
