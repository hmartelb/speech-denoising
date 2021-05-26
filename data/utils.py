import os

import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.functional import magphase
from torchaudio.transforms import Spectrogram


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

    for i, audio in enumerate(audios):
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


def get_magnitude(audio, spectrogram_size=256, mode='amplitude', pad=False, normalize=True, return_phase=False):
    '''
    Compute the magnitude from an audio segment.

    Paramters:
        audio (torch.Tensor):               2D Tensor containing the audio data in the format [n_channels, samples]
        spectrogram_size (int):             Size of the resulting spectrogram after processing.
                                            The Spectrogram transform is computed using n_fft = spectrogram_size * 2 - 1
        mode (str, default: amplitude):     Option to return the magnitude in linear, power of 2, or logarithmic scale (dB).
        pad (bool, default: False):         Pad the time dimension to match the spectrogram_size.
        normalize (bool, default: True):    Apply normalization to the magnitude spectrogram. Divide by 2/spectrogram_size
        return_phase (bool, default: False):Return the phase together with the magnitude
    Returns:
        mag (torch.Tensor):                 Magnitude of the audio spectrogram.
    '''
    n_fft = 2 * spectrogram_size - 1
    S = torchaudio.transforms.Spectrogram(n_fft=n_fft, normalized=normalize, power=(2 if mode == 'power' else None))(audio)

    if mode in ['amplitude', 'db']:
        mag, phase = torchaudio.functional.magphase(S)
    else:
        mag = S # Power spectrogram is directly real-valued

    # if normalize:
    #     mag /= 2*spectrogram_size

    if pad:
        mag = zero_pad(mag, spectrogram_size, 2)

    if mode == 'db':
        mag = torchaudio.transforms.AmplitudeToDB(stype='magnitude')(mag)
    
    if return_phase:
        return mag, phase

    return mag

def get_audio_from_magnitude(mag, phase, spectrogram_size=256, mode='amplitude', normalize=True):
    if mode == 'db':
        mag = torchaudio.functional.DB_to_amplitude(mag) # TODO: check normalization and parameters

    # Undo the padding
    if mag.shape[1] >= phase.shape[1] and mag.shape[2] >= phase.shape[2]:
        mag = mag[:, 0:phase.shape[1], 0:phase.shape[2]]
    
    # S = torch.polar(mag, phase)
    S = mag * torch.exp(1j*phase)

    n_fft = 2 * spectrogram_size - 1
    audio = torch.istft(S, n_fft, window=torch.hann_window(n_fft), hop_length=(n_fft//2 +1), center=True, normalized=normalize, onesided=True, length=None, return_complex=False)
    return audio

def zero_pad(tensor, target_len, dim):
    '''
    Add zeros to a tensor alonng the specified dimension to reach the target length in that dimension.
    '''
    tensor_size = list(tensor.size())
    assert dim <= len(
        tensor_size), "The specified dimension is larger than the tensor rank"
    pad_size = tensor_size
    pad_size[dim] = target_len - tensor_size[dim]
    pad = torch.zeros(pad_size)
    pad_tensor = torch.cat([tensor, pad], dim=dim)
    return pad_tensor


def find_files(directory, extensions=('.mp3', '.wav', '.flac')):
    '''
    Find all the files under a specfied directory (recursively) that one of the allowed extextions.

    Parameters:
        directory (str, path-like):     Root path of the file search
        extensions (tuple):             Extensions that are allowed, the rest of the files will be ignored.
    '''
    for root, dirs, files in os.walk(directory):
        for f in files:
            filename = os.path.join(root, f)
            if filename.endswith(extensions):
                yield filename
