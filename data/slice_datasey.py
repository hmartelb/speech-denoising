import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from utils import print_metadata, find_files


def process_file(input_filename, output_dir, length_seconds=4, pad_last=True):
    try:
        base_path, filename = os.path.split(input_filename)
        name, ext = os.path.splitext(filename)

        audio, sr = torchaudio.load(input_filename)

        segment_length = sr * length_seconds
        n_segments = (audio.shape[1] // segment_length) + \
            (1 if pad_last else 0)

        # Zero pad the last segment if needed
        pad = (n_segments * segment_length) - len(audio)
        if pad > 0:
            audio = torch.cat([audio, torch.zeros([1, pad])], dim=1)

        # Save each segment as {output_dir}/{original_name}_XXXX.{ext}
        for i in range(n_segments):
            audio_segment = audio[:, i*segment_length:(i+1)*segment_length]
            segment_name = os.path.join(
                output_dir, f"{name}_{str(i).zfill(4)}{ext}")
            torchaudio.save(segment_name, audio_segment, sr)

    except Exception as e:
        print_metadata(input_filename)
        print_metadata(output_filename)
        raise e


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--sliced_path', required=True)
    ap.add_argument('--length_seconds', default=4, type=int)
    ap.add_argument('--pad_last', action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(args.sliced_path):
        os.makedirs(args.sliced_path)

    files = list(find_files(args.dataset_path))
    for f_in in tqdm(files):
        f_out = f_in.replace(args.dataset_path, args.sliced_path)

        # Make sure the destination directory exists
        dir_out, _ = os.path.split(f_out)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        # Process the audio file
        process_file(
            f_in, dir_out, length_seconds=args.length_seconds, pad_last=args.pad_last)
