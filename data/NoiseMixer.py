import argparse
import math
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class NoiseMixerDataset(Dataset):
    def __init__(self, clean_dataset, noise_dataset, min_snr=0, max_snr=30):
        self.clean_dataset = clean_dataset
        self.noise_dataset = noise_dataset

        self.min_snr = min_snr
        self.max_snr = max_snr

    def __getitem__(self, index):
        # Use circular indexing (modulo len) so that datasets can have different lengths. Epoch iteration happens over the largest one
        audio, _ = self.clean_dataset[index % len(self.clean_dataset)]
        noise, _ = self.noise_dataset[index % len(self.noise_dataset)]

        audio /= audio.max()
        noise /= noise.max()

        # NOTE: We need to use the PyTorch random number generator (and not Numpy!) to select the random value of SNR.
        # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        snr_db = (self.max_snr-self.min_snr)*torch.rand(1) + self.min_snr

        # Formula to add the background noise from SNR.
        # https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#adding-background-noise
        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / audio_power
        mixture = (scale * audio + noise) / 2
        # mixture = audio + (noise / (10.0**(0.05*snr))) # TODO: Is this equivalent ?

        return mixture, torch.cat([audio, noise], dim=0)

    def __len__(self):
        return max(len(self.clean_dataset), len(self.noise_dataset))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean_dataset_path', required=True)
    ap.add_argument('--noise_dataset_path', required=True)
    ap.add_argument('--target_sr', default=16000, type=int)
    ap.add_argument('--length_seconds', default=4, type=int)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    from LibriSpeech import LibriSpeechDataset
    from UrbanSound8K import UrbanSound8KDataset
    from AudioDirectory import AudioDirectoryDataset

    # train_clean_dataset = LibriSpeechDataset(
    #     args.clean_dataset_path,
    #     split="train-clean-100",
    #     target_sr=args.target_sr,
    #     length_seconds=args.length_seconds
    # )

    # train_noise_dataset = UrbanSound8KDataset(
    #     args.noise_dataset_path,
    #     folders=range(1, 10),
    #     target_sr=args.target_sr,
    #     length_seconds=args.length_seconds
    # )

    train_clean_dataset = AudioDirectoryDataset(root=args.clean_dataset_path)
    train_noise_dataset = AudioDirectoryDataset(root=args.noise_dataset_path)

    train_data = NoiseMixerDataset(
        clean_dataset=train_clean_dataset,
        noise_dataset=train_noise_dataset,
        min_snr=0,
        max_snr=30
    )

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    from tests import iteration_test, shape_test, waveforms_test

    UNIT_TESTS_DIR = os.path.join('unit_tests', 'NoiseMixer')
    if not os.path.isdir(UNIT_TESTS_DIR):
        os.makedirs(UNIT_TESTS_DIR)

    SEGMENT_LENGTH = args.target_sr*args.length_seconds
    shape_test(train_data, [1,SEGMENT_LENGTH], [2,SEGMENT_LENGTH], split='train')
    waveforms_test(
        train_data, 
        sr=args.target_sr, 
        process_output=True,
        plot_filename=os.path.join(UNIT_TESTS_DIR, 'train_single.png'),
        audios_dir=os.path.join(UNIT_TESTS_DIR, 'train_single')    
    )

    shape_test(train_loader, [args.batch_size,1,SEGMENT_LENGTH], [args.batch_size,2,SEGMENT_LENGTH], split='train')
    waveforms_test(
        train_loader, 
        sr=args.target_sr, 
        plot_filename=os.path.join(UNIT_TESTS_DIR, 'train_batch.png'),
        audios_dir=os.path.join(UNIT_TESTS_DIR, 'train_batch')    
    )

    iteration_test(train_loader)
