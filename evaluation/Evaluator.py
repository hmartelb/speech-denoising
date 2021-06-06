import math
import os
from pprint import pprint

import librosa
import numpy as np
import pandas as pd
from pesq import pesq
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

    def eval_single(self, ground_truth, estimation):
        return {
            "SI-SDR": self.SI_SDR(ground_truth, estimation),
            "PESQ-wb": self.PESQ(ground_truth[0,], estimation[0,], mode="wb"),
            "PESQ-nb": self.PESQ(ground_truth[0,], estimation[0,], mode="nb"),
            "STOI": self.STOI(ground_truth[0,], estimation[0,]),
        }

    def eval_directory(self, ground_truth_path, estimations_path):
        stats = {}
        metrics = {"SI-SDR": [], "PESQ-wb": [], "PESQ-nb": [], "STOI": []}

        mixture_files = list(find_files(ground_truth_path, extensions=f"{self.mixture}.wav"))
        for true_file in tqdm(mixture_files):
            estimation_file = true_file.replace(ground_truth_path, estimations_path)

            # Get the names of the sources and load into array of shape [samples, sources]
            true_sources = self._load_sources(
                sources_dict={s: true_file.replace(self.mixture, s) for s in self.sources}, concat=True,
            )
            estimated_sources = self._load_sources(
                sources_dict={s: estimation_file.replace(self.mixture, s) for s in self.sources}, concat=True,
            )

            single_metrics = self.eval_single(true_sources, estimated_sources)
            for sm in single_metrics:
                metrics[sm].append(single_metrics[sm])

        stats = {m: {"mean": np.mean(metrics[m]), "std": np.std(metrics[m])} for m in metrics}
        return stats

    def _load_sources(self, sources_dict, concat=False, fmt="channels_first"):
        loaded = {}
        for s in sources_dict:
            filename = sources_dict[s]
            audio, _ = librosa.load(filename, sr=self.sr)
            audio = np.expand_dims(audio, axis=1)
            loaded.update({s: audio})

        if concat:
            loaded = np.concatenate([loaded[s] for s in loaded], axis=1)
            if fmt == "channels_first":
                loaded = np.transpose(loaded, axes=(1, 0))

        return loaded

    def STOI(self, clean, estimate):
        return stoi(clean, estimate, self.sr, extended=False)

    def PESQ(self, clean, estimate, mode="wb"):
        assert mode in ["wb", "nb"], "Invalid mode, it must be one of ['wb', 'nb']"
        return pesq(self.sr, clean, estimate, mode=mode)

    def SI_SDR(self, clean, estimate, eps=1e-9):
        Rss = np.sum((clean * clean), axis=0)
        a = (eps + np.sum((clean * estimate), axis=0)) / (Rss + eps)
        e_true = a * clean
        e_res = estimate - e_true
        Sss = np.sum((e_true ** 2), axis=0)
        Snn = np.sum((e_res ** 2), axis=0)
        return np.mean(10 * np.log10((eps + Sss) / (eps + Snn)))

    def compute_measures(self, clean, estimate, scaling=True, eps=1e-9):
        Rss = np.sum((clean * clean), axis=0)
        a = (eps + np.sum((clean * estimate), axis=0)) / (Rss + eps) if scaling else 1

        e_true = a * clean
        e_res = estimate - e_true

        Sss = (e_true ** 2).sum()
        Snn = (e_res ** 2).sum()

        sdr = 10 * np.log10(Sss / (Snn + eps))

        Rsr = np.sum((clean * e_res), axis=0)
        # Rsr = np.dot(clean.transpose(), e_res)
        b = np.linalg.solve(Rss, Rsr)

        e_interf = np.dot(clean, b)
        e_artif = e_res - e_interf

        sir = 10 * np.log10(Sss / ((e_interf ** 2).sum() + eps))
        sar = 10 * np.log10(Sss / ((e_artif ** 2).sum() + eps))

        return sdr, sir, sar


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--estimations", required=True)
    args = ap.parse_args()

    assert os.path.isdir(args.ground_truth) and os.path.isdir(args.estimations), "The directories do not exist"

    ev = Evaluator()
    results = ev.eval_directory(args.ground_truth, args.estimations)
    # print(results)

    df = pd.DataFrame(results).T
    print(df.head(10))
    df.to_csv(f"{args.estimations}.csv")