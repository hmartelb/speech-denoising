import os

import torch
import torchaudio
from tqdm import tqdm

from data import AudioDirectoryDataset, NoiseMixerDataset
from data.utils import make_path


def generate_evaluation_data(
    clean_directory, noise_directory, output_directory, min_snr=0, max_snr=18, sr=16000
):
    """
    Generate input and output pais for evaluation
    """

    # Initialize a dataset object
    evaluation_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=clean_directory, keep_rate=1.0),
        noise_dataset=AudioDirectoryDataset(root=noise_directory, keep_rate=1.0),
        mode="time",
        min_snr=min_snr,
        max_snr=max_snr,
    )

    # Get the number of digits to represent the examples
    n_digits = len(str(abs(len(evaluation_data))))

    # Get mixture and sources. Each example is separated in 1 folder
    for i, (mixture, sources) in enumerate(
        tqdm(evaluation_data, total=len(evaluation_data))
    ):
        output_folder = os.path.join(output_directory, str(i).zfill(n_digits))
        make_path(output_folder)

        clean = sources[0:1,]
        noise = sources[1:2,]

        # Save the audios
        torchaudio.save(os.path.join(f"{output_folder}", "mixture.wav"), mixture, sr)
        torchaudio.save(os.path.join(f"{output_folder}", "clean.wav"), clean, sr)
        torchaudio.save(os.path.join(f"{output_folder}", "noise.wav"), noise, sr)

        # This should NOT be necessary, but the loop is not stopping
        if i >= len(evaluation_data):
            break


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--clean_path", required=True)
    ap.add_argument("--noise_path", required=True)
    ap.add_argument("--output_path", required=True)

    # SNR config
    ap.add_argument("--min_snr", default=0)
    ap.add_argument("--max_snr", default=18)

    # Audio config
    ap.add_argument("--sr", default=16000)

    args = ap.parse_args()

    generate_evaluation_data(
        clean_directory=args.clean_path,
        noise_directory=args.noise_path,
        output_directory=args.output_path,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
        sr=args.sr,
    )
