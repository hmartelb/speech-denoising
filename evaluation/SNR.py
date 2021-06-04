import numpy as np
from numpy.core.records import array
from scipy.io import wavfile
import scipy


class SNR(): 
    '''
    inputwavfilepath    <- input audio file
    outputwavfilepath   <- model output audio file
    '''
    def __init__(self, cleanfilepath, noisefilepath):
        clean = wavfile.read(cleanfilepath)[1]
        self.clean = clean
        print(clean, clean.shape)
        noise = wavfile.read(noisefilepath)[1]
        self.noise = noise
        print(noise, noise.shape)
    
    '''
    Ref:
        https://github.com/akueisara/audio-signal-processing/blob/master/week%204/A4/A4Part2.py
    '''
    def energy(self, x):
        e = np.sum(x ** 2)
        return e

    def get_evaluation(self):
        snr = 10 * np.log10(self.energy(self.clean) /
                            self.energy(self.noise))
        return snr

    def signaltonoise(self, array, axis=0, ddof=0):
        array = np.asanyarray(array)
        m = array.mean(axis)
        sd = array.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)

    
    def get_evaluation_single_audio(self):
        singleChannel = self.noise
        try:
            singleChannel = np.sum(self.noise, axis=1)
        except:
            # was mono after all
            pass

        norm = singleChannel / (max(np.amax(singleChannel), -1 * np.amin(singleChannel)))
        snr = self.signaltonoise(norm, axis=0, ddof=0)
        return snr

if __name__ == '__main__':
    path_unet_all_mask_noise = '../results/UNet_all_data_10_epochs_masking/output_noise.wav'
    path_unet_all_mask_clean = '../results/UNet_all_data_10_epochs_masking/output_clean.wav'
    path_unet_10_mask_noise = '../results/UNet_0.1_data_10_epochs_masking/output_noise.wav'
    path_unet_10_nomask_clean = '../results/UNet_0.1_data_10_epochs_masking/output_clean.wav'
    print('SNR unet_all_mask', SNR(path_unet_all_mask_clean, path_unet_all_mask_noise).get_evaluation())
    print('SNR unet_10_mask', SNR(path_unet_10_nomask_clean, path_unet_10_mask_noise).get_evaluation())
