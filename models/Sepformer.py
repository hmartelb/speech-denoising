import torch
import torch.nn as nn
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


class Sepformer(nn.Module):
    def __init__(
        self,
        source="speechbrain/sepformer-whamr16k",
        savedir="tmpdir",
    ):
        super(Sepformer, self).__init__()

        self.source = source
        self.savedir = savedir

        self.model = separator.from_hparams(source=self.source, savedir=self.savedir)

    def forward(self, x):
        return self.model.separate_batch(x.squeeze()).permute(0, 2, 1)


if __name__ == "__main__":
    model = Sepformer()
    x = torch.randn(4, 1, 16000)
    y = model(x)
    print(y.shape)
