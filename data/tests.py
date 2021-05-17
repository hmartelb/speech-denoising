import os

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import plot_waveforms, save_waveforms


def _get_data(dataset):
    input_data, output_data = None, None
    if isinstance(dataset, torch.utils.data.Dataset):
        input_data, output_data = dataset[0]
    elif isinstance(dataset, torch.utils.data.DataLoader):
        for batch in dataset:
            input_data, output_data = batch
            break
    else:
        print("Wrong dataset type")
    return input_data, output_data


def shape_test(dataset, expected_in_shape, expected_out_shape=None, split='train'):
    input_data, output_data = _get_data(dataset)
    assert list(
        input_data.shape) == expected_in_shape, f"Wrong input shape in single example ({split}). Expected {expected_in_shape}, but got {list(input_data.shape)}"
    if expected_out_shape:
        assert list(
            output_data.shape) == expected_out_shape, f"Wrong output shape in single example ({split}). Expected {expected_out_shape}, but got {list(output_data.shape)}"


def waveforms_test(dataset, sr, process_output=False, plot_filename=None, audios_dir=None):
    input_data, output_data = _get_data(dataset)

    if plot_filename:
        plot_waveforms([input_data[i, :]
                       for i in range(input_data.shape[0])], plot_filename)
        if process_output:
            name, ext = os.path.splitext(plot_filename)
            plot_waveforms([output_data[i, :] for i in range(
                output_data.shape[0])], f"{name}_output{ext}")
    if audios_dir:
        save_waveforms([input_data[i, :]
                       for i in range(input_data.shape[0])], sr, audios_dir)
        if process_output:
            save_waveforms([output_data[i, :] for i in range(
                output_data.shape[0])], sr, f'{audios_dir}_output')


def iteration_test(dataset):
    print("Iteration test started")
    if isinstance(dataset, torch.utils.data.DataLoader):
        i = 0
        try:
            for batch in tqdm(dataset, total=len(dataset)):
                i += 1
        except:
            pass
        assert len(dataset) == i, f"Expected {len(dataset)} iterations, but dataset stopped at {i} iterations."