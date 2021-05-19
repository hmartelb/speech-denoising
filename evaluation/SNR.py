import numpy as np
from numpy.core.records import array
from scipy.io import wavfile
import scipy


class SNR(): 
    '''
    inputwavfilepath    <- input audio file
    outputwavfilepath   <- model output audio file
    '''
    def __init__(self, inputwavfilepath, outputwavfilepath):
        samplerate_in, data_in = wavfile.read(inputwavfilepath)
        self.samplerate_in = samplerate_in
        self.data_in = data_in
        samplerate_out, data_out = wavfile.read(outputwavfilepath)
        self.samplerate_out = samplerate_out
        self.data_out = data_out
    
    '''
    Ref:
        https://github.com/akueisara/audio-signal-processing/blob/master/week%204/A4/A4Part2.py
    '''
    def energy(self, x):
        e = np.sum(x ** 2)
        return e

    def get_evaluation(self):
        snr  = 10 * np.log10(self.energy(self.data_in)/self.energy(self.data_in - self.data_out))

    def signaltonoise(self, array, axis=0, ddof=0):
        array = np.asanyarray(array)
        m = array.mean(axis)
        sd = array.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)

    
    def get_evaluation_single_audio(self):
        snr = self.signaltonoise(self.audio_array, axis = 0, ddof = 0)
        print("Sound to Noise Ratio: ", snr)


if __name__ == '__main__':
    inputpath = ''
    outputpath = ''
    snr = SNR(inputpath, outputpath)
    print(snr.get_evaluation())
    
