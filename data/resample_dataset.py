import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from utils import print_metadata, find_files


def process_file(input_filename, output_filename, target_sr):
    try:
        audio, original_sr = torchaudio.load(input_filename)

        # Downmix to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        # Normalize
        audio /= audio.max()

        # Resample from original_sr to target_sr
        audio = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=target_sr)(audio)
        torchaudio.save(output_filename, audio, target_sr)

    except Exception as e:
        print_metadata(input_filename)
        print_metadata(output_filename)
        raise e


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--resampled_path', required=True)
    ap.add_argument('--target_sr', default=16000, type=int)
    args = ap.parse_args()

    if not os.path.isdir(args.resampled_path):
        os.makedirs(args.resampled_path)

    files = list(find_files(args.dataset_path))
    for f_in in tqdm(files):
        f_out = f_in.replace(args.dataset_path, args.resampled_path)

        # Make sure the destination directory exists
        dir_out, _ = os.path.split(f_out)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        # Process the audio file
        process_file(f_in, f_out, target_sr=args.target_sr)
