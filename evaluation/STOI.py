import numpy as np
import os
from scipy.io import wavfile
import scipy



class STOI():
    def __init__(self, estimate, clean):
        self.estimate = estimate
        self.clean = clean

    def get_evaluation(self, rate):
        from pystoi.stoi import stoi
        return stoi(self.clean, self.estimate, rate, extended=False)

if __name__ == '__main__':
    #given as an input to the model
    true_src_list = []
    true_noise_list = []
    true_mixture_list = []
    #output of the model
    estimated_src_list = []
    estimated_noise_list = []

    # List all subdirectories using scandir()
    basepath = './test_data/'

    for dirpath, dirnames, files in os.walk(basepath):
        for dirname in dirnames:

            if dirname == "ground_truth":
                basepath_2 = os.path.join(dirpath, dirname)
                print(basepath_2)
                for dirpath_2, dirnames_2, files_2 in os.walk(basepath_2):
                    for dirname_2 in dirnames_2:
                        basepath_3 = os.path.join(basepath_2, dirname_2)
                        print(basepath_3)
                        with os.scandir(basepath_3) as entries:
                            for entry in entries:
                                file_path = os.path.join(
                                    basepath_3, entry.name)
                                rate = wavfile.read(file_path)[0]
                                data = wavfile.read(file_path)[1]

                                if entry.name == 'clean.wav':
                                    print("clean rate", rate)
                                    true_src_list.append(data)

                                if entry.name == 'mixture.wav':
                                    print("mixture rate", rate)
                                    true_noise_list.append(data)

                                if entry.name == 'noise.wav':
                                    print("noise rate", rate)
                                    true_mixture_list.append(data)

            if dirname == "UNetDNP_SI-SDR_10_epochs":
                basepath_2 = os.path.join(dirpath, dirname)
                print(basepath_2)
                for dirpath_2, dirnames_2, files_2 in os.walk(basepath_2):
                    for dirname_2 in dirnames_2:
                        basepath_3 = os.path.join(basepath_2, dirname_2)
                        print(basepath_3)
                        with os.scandir(basepath_3) as entries:
                            for entry in entries:
                                file_path = os.path.join(
                                    basepath_3, entry.name)
                                rate = wavfile.read(file_path)[0]
                                data = wavfile.read(file_path)[1]

                                if entry.name == 'clean.wav':
                                    print("est clean rate", rate)
                                    estimated_src_list.append(data)

                                if entry.name == 'noise.wav':
                                    print("est noise rate", rate)
                                    estimated_noise_list.append(data)

    #convert to numpy array
    true_src_list = np.asarray(true_src_list)
    true_noise_list = np.asarray(true_noise_list)
    true_mixture_list = np.asarray(true_mixture_list)
    estimated_src_list = np.asarray(estimated_src_list)
    estimated_noise_list = np.asarray(estimated_noise_list)
    #get the evaluation
    stoi_list = []

    for clean, estimate in zip(true_src_list, estimated_src_list):
        stoi = STOI(estimate, clean).get_evaluation(16000)
        stoi_list.append(stoi)
    
    print(np.mean(stoi_list))
