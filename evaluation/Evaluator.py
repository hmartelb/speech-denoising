import math
import os
from pprint import pprint

import librosa
import numpy as np
import pandas as pd
from pesq import pesq, PesqError
from pystoi.stoi import stoi

from tqdm import tqdm


def find_files(directory, extensions=(".mp3", ".wav", ".flac")):
    """
    Find all the files under a specfied directory (recursively) that one of the allowed extextions.

    Parameters:
        directory (str, path-like):     Root path of the file search
        extensions (tuple):             Extensions that are allowed, the rest of the files will be ignored.
    """
    for root, dirs, files in os.walk(directory):
        for f in files:
            filename = os.path.join(root, f)
            if filename.endswith(extensions):
                yield filename


class Evaluator:
    def __init__(
        self, metrics={}, mixture="mixture", sources=["clean", "noise"], sr=16000,
    ):
        self.mixture = mixture
        self.sources = sources
        self.sr = sr

        self.pesq_wb_errors = 0
        self.pesq_nb_errors = 0

    def eval_single(self, ground_truth, estimation, idx):
        sdr, sir, sar = self.blind_sep_metrics(ground_truth, estimation, scaling=True)
        # sdr, sir, sar = 0,0,0
        pesq_wb = self.PESQ(ground_truth[:,0], estimation[:,0], mode="wb")
        if pesq_wb < -0.999:
            self.pesq_wb_errors += 1
        #     print(f"Error wb idx={idx}, value={pesq_wb}")
        pesq_nb = self.PESQ(ground_truth[:,0], estimation[:,0], mode="nb")
        if pesq_nb < -0.999:
            self.pesq_nb_errors += 1
        #     print(f"Error nb idx={idx}, value={pesq_nb}")
        stoi = self.STOI(ground_truth[:,0], estimation[:,0])
        # stoi = 0

        return {
            "SDR": sdr,
            "SIR": sir, 
            "SAR": sar,
            "PESQ-wb": pesq_wb,
            "PESQ-nb": pesq_nb,
            "STOI": stoi,
        }

    def eval_directory(self, ground_truth_path, estimations_path):
        # stats = {}
        metrics = { "SDR": [], "SIR": [], "SAR": [], "PESQ-wb": [], "PESQ-nb": [], "STOI": [] }
        failures = 0

        mixture_files = list(find_files(ground_truth_path, extensions=f"{self.mixture}.wav"))
        
        for i,true_file in enumerate(tqdm(mixture_files)):
            estimation_file = true_file.replace(ground_truth_path, estimations_path)

            # Get the names of the sources and load into array of shape [samples, sources]
            true_sources = self._load_sources(
                sources_dict={s: true_file.replace(self.mixture, s) for s in self.sources}, concat=True,
            )
            estimated_sources = self._load_sources(
                sources_dict={s: estimation_file.replace(self.mixture, s) for s in self.sources}, concat=True,
            )

            # try:
            
            single_metrics = self.eval_single(true_sources, estimated_sources, idx=i)
            for sm in single_metrics:
                metrics[sm].append(single_metrics[sm])
            # except:
            #     failures += 1

       

        print(f"Evaluation finished")
        print(f"- Directories (gt, pred): {ground_truth_path, estimations_path}")
        print(f"- Total files: {len(mixture_files)}")
        print(f"- Failures: {failures}")

        return metrics
        # stats = {m: {"mean": np.mean(metrics[m]), "std": np.std(metrics[m])} for m in metrics}
        # return stats

    def _load_sources(self, sources_dict, concat=False, fmt="channels_first"):
        loaded = {}
        for s in sources_dict:
            filename = sources_dict[s]
            audio, _ = librosa.load(filename, sr=self.sr)
            audio = np.expand_dims(audio, axis=1)
            loaded.update({s: audio})

        if concat:
            loaded = np.concatenate([loaded[s] for s in loaded], axis=1)
            # if fmt == "channels_first":
            #     loaded = np.transpose(loaded, axes=(1, 0))

        return loaded

    def STOI(self, clean, estimate):
        return stoi(clean, estimate, self.sr, extended=False)

    def PESQ(self, clean, estimate, mode="wb"):
        assert mode in ["wb", "nb"], "Invalid mode, it must be one of ['wb', 'nb']"
        return pesq(self.sr, clean, estimate, mode=mode, on_error=PesqError.RETURN_VALUES)

    def blind_sep_metrics(self, reference, estimate, scaling=True):
        reference -= np.mean(reference, axis=0)
        estimate -= np.mean(estimate, axis=0)

        eps = np.finfo(estimate[0].dtype).eps

        # SDR
        Rss = np.sum((reference * reference), axis=0)

        scale = (eps + np.sum((reference * estimate), axis=0)) / (Rss + eps) if scaling else 1

        e_true = scale * reference
        e_res = estimate - e_true

        Sss = np.sum((e_true**2), axis=0)
        Snn = np.sum((e_res**2), axis=0)

        # SIR and SAR
        Rsr = np.dot(reference.T, e_res)
        Rss = np.dot(reference.T, reference) ## FIXME: Is this needed? Check with np.sum((reference * reference), axis=0)

        b = np.linalg.lstsq(Rss, Rsr)
        e_interf = np.dot(reference, b[0])
        e_artif = e_res - e_interf

        SDR = np.mean(10 * np.log10((Sss + eps) / (eps + Snn)))
        SIR = np.mean(10 * np.log10((Sss + eps) / (eps + np.sum(e_interf**2, axis=0))))
        SAR = np.mean(10 * np.log10((Sss + eps) / (eps + np.sum(e_artif**2, axis=0))))      

        return SDR, SIR, SAR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--estimations", required=True)
    args = ap.parse_args()

    assert os.path.isdir(args.ground_truth) and os.path.isdir(args.estimations), "The directories do not exist"

    ev = Evaluator()
    results = ev.eval_directory(args.ground_truth, args.estimations)

    print(ev.pesq_wb_errors, ev.pesq_nb_errors)

    df = pd.DataFrame(results)
    print(df.head(10))
    print(df.describe())
    df.to_csv(f"{args.estimations}.csv")