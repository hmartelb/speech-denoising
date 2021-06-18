import argparse
import torch
import torchaudio
import os
import numpy as np
from data.utils import get_magnitude, get_audio_from_magnitude
import matplotlib.pyplot as plt

from getmodel import get_model


def predict_spectrogram(audio, sr, length_seconds, model):
    total_samples = audio.shape[1]
    segment_length = sr * length_seconds
    n_segments = int(np.ceil(audio.shape[1] / segment_length))

    output_segments = {"clean": [], "noise": []}
    for i in range(n_segments):
        # print(f"Processing segment {i+1}/{n_segments}")
        if audio.shape[1] >= (i + 1) * segment_length:
            seg_audio = audio[:, i * segment_length : (i + 1) * segment_length]
        else:
            seg_audio = torch.zeros([1, segment_length])
            seg_audio[:, 0 : audio.shape[1] - i * segment_length] = audio[:, i * segment_length :]

        # Forward transform for the input mixture
        # 1) Compute the STFT spectrogram and the phase
        # 2) Add batch dimension
        # 3) Get predictions from the model
        seg_magnitude, seg_phase = get_magnitude(
            seg_audio, spectrogram_size=256, mode="amplitude", normalize=False, pad=True, return_phase=True
        )
        seg_magnitude = seg_magnitude.unsqueeze(0)  # Add batch dimension
        out_magnitude = model(seg_magnitude.cuda())  # Use the model
        out_magnitude = out_magnitude.squeeze()  # Remove batch dimension
        out_magnitude = out_magnitude.cpu().detach()

        # Inverse transform for each source
        # 1) Get the corresponding channel from the output
        # 2) Apply the inverse transform to recover audio from spectrogram
        # 3) Trim to the original length (ISTFT may not output the original number of samples)
        #
        clean_magnitude = out_magnitude[0:1, :, :]
        clean_audio = get_audio_from_magnitude(
            clean_magnitude, seg_phase, spectrogram_size=256, mode="amplitude", normalize=False
        )
        clean_audio = clean_audio[:, 0:segment_length]

        noise_magnitude = out_magnitude[1:2, :, :]
        noise_audio = get_audio_from_magnitude(
            noise_magnitude, seg_phase, spectrogram_size=256, mode="amplitude", normalize=False
        )
        noise_audio = noise_audio[:, 0:segment_length]

        # Append the obtained segments for each source into a list
        output_segments["clean"].append(clean_audio)
        output_segments["noise"].append(noise_audio)

    # Concatenate along time dimension to obtain the full audio
    clean_output = torch.cat(output_segments["clean"], dim=1)
    noise_output = torch.cat(output_segments["noise"], dim=1)

    return clean_output[:, 0:total_samples], noise_output[:, 0:total_samples]


def predict_waveform(audio, sr, length_seconds, model):
    total_samples = audio.shape[1]
    segment_length = sr * length_seconds
    n_segments = int(np.ceil(audio.shape[1] / segment_length))

    output_segments = {"clean": [], "noise": []}
    for i in range(n_segments):
        # print(f"Processing segment {i+1}/{n_segments}")
        if audio.shape[1] >= (i + 1) * segment_length:
            seg_audio = audio[:, i * segment_length : (i + 1) * segment_length]
        else:
            seg_audio = torch.zeros([1, segment_length])
            seg_audio[:, 0 : audio.shape[1] - i * segment_length] = audio[:, i * segment_length :]

        seg_audio = seg_audio.unsqueeze(0)
        out_sources = model(seg_audio)  # Use the model
        out_sources = out_sources.squeeze()
        out_sources = out_sources.cpu().detach()

        clean_audio = out_sources[0:1, :]
        noise_audio = out_sources[1:2, :]

        # Append the obtained segments for each source into a list
        output_segments["clean"].append(clean_audio)
        output_segments["noise"].append(noise_audio)

    # Concatenate along time dimension to obtain the full audio
    clean_output = torch.cat(output_segments["clean"], dim=1)
    noise_output = torch.cat(output_segments["noise"], dim=1)

    return clean_output[:, 0:total_samples], noise_output[:, 0:total_samples]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Input and Output
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)

    # Model to use
    ap.add_argument("--model", choices=["UNet", "UNetDNP", "ConvTasNet", "TransUNet", "SepFormer"])
    ap.add_argument("--checkpoint_name", required=True, help="File with .tar extension")

    # Data parameters
    ap.add_argument("--length_seconds", default=4, type=int)
    # ap.add_argument("--mode", default="amplitude")

    # GPU setup
    ap.add_argument("--gpu", default="-1")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    visible_devices = list(map(lambda x: int(x), args.gpu.split(",")))
    print("Visible devices:", visible_devices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")


    # Get the model
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

    extensions = (".mp3", ".wav", ".flac")
    assert os.path.isfile(args.input) and args.input.endswith(
        extensions
    ), f"Input file cannot be loaded. Either it does not exist or has a wrong extension. Allowed extensions {extensions}"

    audio, sr = torchaudio.load(args.input)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000

    audio /= audio.abs().max()

    if data_mode in ["time"]:
        clean_output, noise_output = predict_waveform(audio.to(device), sr, args.length_seconds, model)

    if data_mode in ["amplitude", "power", "db"]:
        clean_output, noise_output = predict_spectrogram(audio, sr, args.length_seconds, model)

    # Normalization wrt mixture
    clean_output /= audio.abs().max()#clean_output.abs().max()
    noise_output /= audio.abs().max()#noise_output.abs().max()

    plt.subplot(3, 1, 1)
    plt.plot(
        audio[
            0,
        ]
    )
    plt.subplot(3, 1, 2)
    plt.plot(
        clean_output[
            0,
        ]
    )
    plt.subplot(3, 1, 3)
    plt.plot(
        noise_output[
            0,
        ]
    )
    plt.show()

    output_name, ext = os.path.splitext(args.output)

    torchaudio.save(f"{output_name}_clean{ext}", clean_output, sr)
    torchaudio.save(f"{output_name}_noise{ext}", noise_output, sr)
