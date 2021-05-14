# Audio Denoising

Audio Denoising project for the Deep Learning course at Tsinghua University, Spring semester 2021.

## Code organization

```
/data: "Functions to load the datasets and generate data augmentations."
    __init__.py
    LibriSpeech.py
    NoiseMixer.py
    tests.py
    UrbanSound8K.py
    utils.py

/evaluation: "Common metrics to evaluate the results"
    __init__.py

/models: "Model architectures. Each model is in one Python file."
    __init__.py
```

## Quick start

### Environment setup
Clone this repository to your system.
```
$ git clone https://github.com/hmartelb/audio-denoising.git
```

Make sure that you have Python 3 installed in your system. Also, Pytorch 1.5 or above needs to be installed. Check the [official installation guide](https://pytorch.org/get-started/locally/) to set it up according to your system requirements and CUDA version.

It is recommended to create a virtual environment to install the dependencies. Open a new terminal in the master directory, activate the virtual environment and install the dependencies from requirements.txt by executing this command:

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
* Example of data loader (Kaggle): [https://www.kaggle.com/ronyroy/torchaudio-urban-sounds-8k](https://www.kaggle.com/ronyroy/torchaudio-urban-sounds-8k) 

In addition, the following audio augmentation packages are used: 
* 
* WavAugment (Facebook): [https://github.com/facebookresearch/WavAugment](https://github.com/facebookresearch/WavAugment)

## Evaluation

speechmetrics: [https://github.com/aliutkus/speechmetrics](https://github.com/aliutkus/speechmetrics)

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