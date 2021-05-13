import os

import matplotlib.pyplot as plt
import torchaudio
import torch


def plot_waveforms(audios, to_file=None):
    '''
    Display the provided audios in time domain, using subplots.

    Input parameters:
    - audios (list, torch.tensor, np.array): Data to be displayed.

    '''
    if type(audios) != list:
        audios = [audios]

    num_plots = len(audios)
    plt.figure()
    for i, audio in enumerate(audios):
        if torch.is_tensor(audio):
            audio = audio.squeeze().cpu().numpy()

        plt.subplot(num_plots, 1, i+1)
        plt.plot(audio)

    if to_file is None:
        plt.show()
    else:
        plt.savefig(to_file, dpi=300)

def save_waveforms(audios, sr, base_dir):
    '''
    Save the provided audios as .wav in base_dir.

    Input parameters:
    - audios (list, torch.tensor, np.array): Audio data.
    - sr (int): Sampling rate of the audios.
    - base_dir (str, path): Directory to which save the files (will be created if it does not exist).
    '''

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    
    if type(audios) != list:
        audios = [audios]

    for i,audio in enumerate(audios):
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        filename = os.path.join(base_dir, f"{i}.wav")
        torchaudio.save(filename, audio, sr)

def print_metadata(audio_file):
    '''
    Show the metadata of an audio file given its path.
    Inspired from the Torchaudio preprocessing tutorial:
    https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

    Input parameters:
    - audio_file (str, path): Path to the file to be analyzed.

    '''
    metadata = torchaudio.info(audio_file)
    print("-" * 10)
    print("Source:", audio_file)
    print("-" * 10)
    print(" - sample_rate:", metadata.sample_rate)
    print(" - num_channels:", metadata.num_channels)
    print(" - num_frames:", metadata.num_frames)
    print(" - bits_per_sample:", metadata.bits_per_sample)
    print(" - encoding:", metadata.encoding)
    print()
