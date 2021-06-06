from .losses import ScaleInvariantSDRLoss
from .models import *
import torch.functional as F


def get_model(name, parameters):
    if name == "UNet":
        model = UNet(**parameters)
        data_mode = "amplitude"
        loss_fn = F.mse_loss
        loss_mode = "min"

    if name == "UNetDNP":
        model = UNetDNP(**parameters)
        data_mode = "time"
        loss_fn = ScaleInvariantSDRLoss
        loss_mode = "max"

    if name == "ConvTasNet":
        model = ConvTasNet(**parameters)
        data_mode = "time"
        loss_fn = ScaleInvariantSDRLoss
        loss_mode = "max"

    if name == "TransUNet":
        model = TransUNet(**parameters)
        data_mode = "amplitude"
        loss_fn = F.mse_loss
        loss_mode = "min"

    if name == "SepFormer":
        model = Sepformer(**parameters)
        data_mode = "time"
        loss_fn = ScaleInvariantSDRLoss
        loss_mode = "max"

    return {
        "model": model,
        "data_mode": data_mode,
        "loss_fn": loss_fn,
        "loss_mode": loss_mode,
    }
