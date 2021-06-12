import argparse
import math
import os

import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import get_magnitude, zero_pad


class NoiseMixerDataset(Dataset):
    def __init__(self, clean_dataset, noise_dataset, min_snr=0, max_snr=18, mode='time', spectrogram_size=256, eps=1e-6):
        '''
        Mix 2 datasets containing clean audios and background noise. 
        The iteration is defined as the largest dataset, in modulo length of each.

        Parameters:
            clean_dataset (Dataset object):                     Dataset containing the clean audios.
            noise_dataset (Dataset object):                     Dataset containing the background noises.
            [Optional] min_snr (int, default: 0):               Minimum SNR value (dB).
            [Optional] max_snr (int, default: 30):              Maximum SNR value (dB).
            [Optional] mode (str, default: 'time'):             Specify to return the time domain signal ('time') or the spectrograms ('amplitude', 'power', 'db').     
            [Optional] spectorgram_size (int, default: 256):    Size of the spectrogram along time and frequency dimensions.      
        '''
        self.clean_dataset = clean_dataset
        self.noise_dataset = noise_dataset

        '''
        Signal above noise: [min_snr: 0dB, max_snr: 18dB]
        Noise can be louder than signal: [min_snr: -12dB, max_snr: 18dB] (Significantly more challenging)

        For the range of SNR, refer to the review paper: "Supervised Speech Separation Based on Deep Learning: An Overview" (Figure 1)
        https://arxiv.org/ftp/arxiv/papers/1708/1708.07524.pdf
        '''
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.eps = eps

        assert mode in ['time', 'amplitude', 'power',
                        'db'], "Invalid mode, it must be one of: ['time', 'amplitude', 'power', 'db']"
        self.mode = mode
        self.spectrogram_size = spectrogram_size

    def __getitem__(self, index):
        # Use circular indexing (modulo len) so that datasets can have different lengths. Epoch iteration happens over the largest one
        clean, _ = self.clean_dataset[index % len(self.clean_dataset)]
        noise, _ = self.noise_dataset[index % len(self.noise_dataset)]

        # NOTE: We need to use the PyTorch random number generator (and not Numpy!) to select the random value of SNR.
        # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        snr_db = (self.max_snr-self.min_snr)*torch.rand(1) + self.min_snr 
        snr_db += self.eps

        # Formula to add the background noise from SNR.
        # https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#adding-background-noise
        clean_power = clean.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / clean_power

        clean *= scale
        mixture = clean + noise

        # Get the max value across all sources and mixture for normalization
        norm_factor = torch.max(torch.max(mixture.abs().max(), clean.abs().max()), noise.abs().max())

        # print(mixture.abs().max(), clean.abs().max(), noise.abs().max())
        # print(norm_factor)
    
        # Normalize wrt mixture
        mixture /= norm_factor
        clean /= norm_factor
        noise /= norm_factor

        if self.mode in ['amplitude', 'power', 'db']:
            # Compute the magnitude spectrogram
            mixture = get_magnitude(mixture, self.spectrogram_size, self.mode, pad=True, normalize=True)
            clean = get_magnitude(clean, self.spectrogram_size, self.mode, pad=True, normalize=True)
            noise = get_magnitude(noise, self.spectrogram_size, self.mode, pad=True, normalize=True)

        return mixture, torch.cat([clean, noise], dim=0)

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

    from .AudioDirectory import AudioDirectoryDataset
    # from LibriSpeech import LibriSpeechDataset
    # from UrbanSound8K import UrbanSound8KDataset

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
    shape_test(train_data, [1, SEGMENT_LENGTH], [
               2, SEGMENT_LENGTH], split='train')
    waveforms_test(
        train_data,
        sr=args.target_sr,
        process_output=True,
        plot_filename=os.path.join(UNIT_TESTS_DIR, 'train_single.png'),
        audios_dir=os.path.join(UNIT_TESTS_DIR, 'train_single')
    )

    shape_test(train_loader, [args.batch_size, 1, SEGMENT_LENGTH], [
               args.batch_size, 2, SEGMENT_LENGTH], split='train')
    waveforms_test(
        train_loader,
        sr=args.target_sr,
        plot_filename=os.path.join(UNIT_TESTS_DIR, 'train_batch.png'),
        audios_dir=os.path.join(UNIT_TESTS_DIR, 'train_batch')
    )

    iteration_test(train_loader)
