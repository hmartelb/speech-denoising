import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from utils import print_metadata, find_files


def process_file(f):
    '''
    Simple but effective method to remove audio files that are silence.
    Check the sum of the absolute value of the Tensor, delete the file if its sum is 0.
    '''
    try:
        audio, original_sr = torchaudio.load(f)
        if audio.abs().sum() == 0:
            os.remove(f)
            return 1
        return 0

    except Exception as e:
        print_metadata(f)
        raise e


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    args = ap.parse_args()

    files = list(find_files(args.dataset_path))
    total_removed = 0
    for f in tqdm(files):
        total_removed += process_file(f)

    print(f"Removed {total_removed} files from {args.dataset_path}")