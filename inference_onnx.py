import argparse
import os

import numpy as np
import onnxruntime
import torchaudio
import torch 

SAMPLING_RATE = 16000
LENGTH_SECONDS = 4


def preprocess_audio(audio, total_samples):
    # total_samples = audio.shape[1]
    segment_length = SAMPLING_RATE * LENGTH_SECONDS
    n_segments = int(np.ceil(audio.shape[1] / segment_length))

    # Apply zero padding to the end if necessary
    samples_to_pad = n_segments * segment_length - total_samples
    if samples_to_pad > 0:
        zeros = torch.zeros(1, samples_to_pad)
        audio = torch.cat([audio, zeros], dim=0)
    
    # Reshape as batch [n_segments, 1, SAMPLING_RATE * LENGTH_SECONDS]
    audio = torch.reshape(audio, (n_segments, 1, segment_length))
    return audio


def postprocess_audio(audio, total_samples):
    n_segments = audio.shape[0]
    segment_length = audio.shape[2]
    
    # Select only the clean speech (channel 0)
    audio = audio[:, 0:1, :]

    # Apply a reshape to convert back to continuous audio
    audio = torch.reshape(audio, (1, segment_length * n_segments))
    audio = audio[:, 0:total_samples]
    return audio


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    assert os.path.isfile(args.model) and args.model.endswith(
        ".onnx"
    ), "ERROR: The provided ONNX argument is incorrect"

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Load and preprocess the audio
    audio_tensor, _ = torchaudio.load(args.input)
    total_samples = audio_tensor.shape[1]
    print(audio_tensor.shape)
    audio_tensor = preprocess_audio(audio_tensor, total_samples)
    print(audio_tensor.shape)

    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(audio_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0]

    print(ort_outs)

    # Apply postprocessing and save
    ort_outs = torch.from_numpy(ort_outs)
    print(ort_outs.shape)
    output_tensor = postprocess_audio(ort_outs, total_samples)

    output_tensor /= output_tensor.abs().max()
    print(output_tensor.shape)
    torchaudio.save(args.output, output_tensor, SAMPLING_RATE)