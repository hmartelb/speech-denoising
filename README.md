# Speech Denoising

Speech Denoising project for the Deep Learning course at Tsinghua University, Spring semester 2021.

![Alt text](docs\img\Speech_denoising_cover.PNG?raw=true "Title")

This code uses a Source Separation approach to recover clean speech signals from a noisy acoustic environment. The high diversity of noises in the dataset motivated to perform the optimization on 2 sources, namely the clean speech signal and the background noise. 

We study 4 different model architectures (2 for time domain and 2 for frequency domain), and compare their performance using Source Separation metrics (SDR, SIR, SAR) and Speech Quality metrics (PESQ, STOI).

## Code organization

```
/data: "Functions to load the datasets and generate data augmentations."
    __init__.py
    AudioDirectory.py       "Dataloader for a directory of audios"
    NoiseMixer.py           "Dataloader to generate clean + noise audio mixtures"
    resample_dataset.py     "Set sampling rate on a directory"
    slice_dataset.py        "Slice all audios in the dataset to equal lengths"
    tests.py                
    utils.py                "Functions to plot, load and save audios, compute STFT"

/evaluation: "Common metrics to evaluate the results and Evaluator class"
    __init__.py
    Evaluator.py    "Class to compute all the evaluation metrics on a directory"
    PESQ.py
    SDR.py
    SI_SAR.py
    SI_SDR.py
    SI_SIR.py
    STOI.py

/losses: "Custom loss functions"
    __init__.py
    SI_SDR.py       "Loss for time domain models"
    STFT.py         "Losses for time and frequency domain, considering spectrograms"

/models: "Model architectures. Each model is in one Python file"
    __init__.py
    Sepformer.py 
    TransUNet.py
    UNet.py
    UNetDNP.py
    
evaluate.py         "Generate audios to perform the evaluation"
getmodel.py         "Build a model with the given parameters"
predict.py          "Perform inference on single audio files"
run_experiments.py  "Train all the models in 1 script"
trainer.py          "Class to train a given model"
```

## Quick start

### Environment setup
Clone this repository to your system.
```
$ git clone https://github.com/hmartelb/speech-denoising.git
```

Make sure that you have Python 3 installed in your system. Also, Pytorch 1.5 or above needs to be installed. Check the [official installation guide](https://pytorch.org/get-started/locally/) to set it up according to your system requirements and CUDA version.

It is recommended to create a virtual environment to install the dependencies. Open a new terminal in the master directory, activate the virtual environment and install the dependencies from ``requirements.txt`` by executing this command:

```
$ (venv) pip install -r requirements.txt
```

## Datasets

We use datasets of clean speech data and noise, which are combined to produce the training data for our denoising models. 

### LibriSpeech (Speech data)
* Dataset download: [http://www.openslr.org/12](http://www.openslr.org/12)
* Torchaudio documentation: [https://pytorch.org/audio/stable/datasets.html#librispeech](https://pytorch.org/audio/stable/datasets.html#librispeech)

### UrbanSounds8K (Noise data)
* Homepage: [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)
* Dataset download: [https://zenodo.org/record/401395#.YJwKzuhKiUk](https://zenodo.org/record/401395#.YJwKzuhKiUk)

### Prepare the data

After downloading the UrbanSound8K we need to preprocess it using the ``resample_dataset.py`` script. This script generates a precomputed version of the dataset with the target sampling rate (default: 16kHz) and downmixed to mono, to speed up the data loading. **You may need to copy the /metadata folder by hand in the resampled dataset**. Usage example:
```
(venv) python resample_dataset.py --dataset_path <path-to-UrbanSound8K>
                                  --resampled_path <path-to-UrbanSound8K_16kHz>
                                  --target_sr 16000
```

Then, slice the LibriSpeech dataset and the UrbanSound8K dataset using the script ``slice_dataset.py``. Add the ``--pad_last`` flag to preserve segments shorter than the give segment length. Usage example:
```
(venv) python slice_dataset.py  --dataset_path <path-to-dataset_16kHz>
                                --sliced_path <path-to-dataset_16kHz_sliced>
                                --length_seconds 4
                                [--pad_last] 
```

## Train the models
The easiest way to train several models is to execute ``run_experiments.py``. Alternatively, a single model can be trained using ``trainer.py``. Please refer to each python file for more details about the command line arguments.
```
(venv) python run_experiments.py    --clean_train_path <path-to-ground-truth-data>
                                    --clean_val_path <path-to-predicted-data>
                                    --noise_train_path <name-of-the-model>
                                    --noise_val_path <filename-of-checkpoint>
                                    [--keep_rate 1 (float between 0 and 1)]
                                    [--gpu <device-id> (defaults to -1: CPU)] 
```

## Evaluation
The evaluation is performed on a set of *ground truth* synthetic mixtures, so the first step is to generate them. Then, we can use any model to obtain the estimation from those mixtures. To do so, run the following command (the first time, the *ground truth* mixtures will be generated in the specified path):
```
(venv) python evaluate.py   --evaluation_path <path-to-ground-truth-data>
                            --output_path <path-to-predicted-data>
                            --model <name-of-the-model>
                            --checkpoint_name <filename-of-checkpoint>
                            [--gpu <device-id> (defaults to -1: CPU)] 
```

After that, run the Evaluator on the directories to obtain the objective metrics:
```
(venv) python evaluation/Evaluator.py   --ground_truth <path-to-ground-truth-data>
                                        --estimations <path-to-predicted-data>
```

## License 
```
MIT License

Copyright (c) 2021 Héctor Martel, Samuel Pegg, Toghrul Abbasli
Master in Advanced Computing, Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
