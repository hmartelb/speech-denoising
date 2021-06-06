import os

import torch
import torchaudio
from tqdm import tqdm

from data import AudioDirectoryDataset, NoiseMixerDataset
from data.utils import find_files, make_path
from getmodel import get_model

# from models import *
from predict import predict_spectrogram, predict_waveform


def predict_evaluation_data(
    evaluation_directory, output_directory, model, data_mode="time", length_seconds=4, normalize=False,
):
    mixture_filenames = [f for f in find_files(evaluation_directory) if f.endswith("mixture.wav")]

    for f in tqdm(mixture_filenames):
        mixture, sr = torchaudio.load(f)
        if sr != 16000:
            mixture = torchaudio.transforms.Resample(sr, 16000)(mixture)
            sr = 16000

        mixture /= mixture.abs().max()
        mixture = mixture.cuda()

        if data_mode == "time":
            clean_output, noise_output = predict_waveform(mixture, sr, length_seconds, model)
        else:
            clean_output, noise_output = predict_spectrogram(mixture, sr, length_seconds, model)

        if normalize:
            clean_output /= clean_output.abs().max()
            noise_output /= noise_output.abs().max()

        # Generate the output names
        clean_output_filename = f.replace(evaluation_directory, output_directory).replace("mixture", "clean")
        noise_output_filename = f.replace(evaluation_directory, output_directory).replace("mixture", "noise")

        output_folder, _ = os.path.split(clean_output_filename)
        make_path(output_folder)

        # Save the audios
        torchaudio.save(clean_output_filename, clean_output, sr)
        torchaudio.save(noise_output_filename, noise_output, sr)


def generate_evaluation_data(
    clean_directory, noise_directory, output_directory, min_snr=0, max_snr=18, sr=16000,
):
    """
    Generate input and output pais for evaluation
    """
    # Create the output directory if it does not exist
    make_path(output_directory)

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
    for i, (mixture, sources) in enumerate(tqdm(evaluation_data, total=len(evaluation_data))):
        output_folder = os.path.join(output_directory, str(i).zfill(n_digits))
        make_path(output_folder)

        clean = sources[
            0:1,
        ]
        noise = sources[
            1:2,
        ]

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
    ap.add_argument("--evaluation_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--clean_path", required=False)
    ap.add_argument("--noise_path", required=False)

    # SNR config
    ap.add_argument("--min_snr", default=0)
    ap.add_argument("--max_snr", default=18)

    # Audio config
    ap.add_argument("--sr", default=16000)

    # Model to use
    ap.add_argument("--model", choices=["UNet", "UNetDNP", "ConvTasNet", "TransUNet", "SepFormer"])
    ap.add_argument("--checkpoint_name", required=True, help="File with .tar extension")

    # GPU setup
    ap.add_argument("--gpu", default="-1")

    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    if not os.path.isdir(args.evaluation_path):
        print("-" * 50)
        print(f"Generating data for evaluation in: {args.evaluation_path}")
        print(f"- Clean audios: {args.clean_path}")
        print(f"- Noise audios: {args.noise_path}")
        print("-" * 50)
        make_path(args.evaluation_path)

        assert os.path.isdir(args.clean_path), f"Path not found {args.clean_path}"
        assert os.path.isdir(args.noise_path), f"Path not found {args.noise_path}"

        generate_evaluation_data(
            clean_directory=args.clean_path,
            noise_directory=args.noise_path,
            output_directory=args.evaluation_path,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            sr=args.sr,
        )

    # Get the model and the data mode
    training_utils_dict = get_model(args.model)

    model = training_utils_dict["model"]
    data_mode = training_utils_dict["data_mode"]
    # loss_fn = training_utils_dict["loss_fn"]
    # loss_mode = training_utils_dict["loss_mode"]

    assert os.path.isfile(args.checkpoint_name) and args.checkpoint_name.endswith(
        ".tar"
    ), "The specified checkpoint_name is not a valid checkpoint"
    checkpoint = torch.load(args.checkpoint_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from checkpoint: {args.checkpoint_name}")

    predict_evaluation_data(
        evaluation_directory=args.evaluation_path,
        output_directory=args.output_path,
        model=model,
        data_mode=data_mode,
        length_seconds=4,
        normalize=True,
    )
