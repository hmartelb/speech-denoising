import argparse
import os

import onnx
import torch

from getmodel import get_model
import numpy as np
import onnxruntime


def load_architecture(model_name):
    model = get_model(args.model)["model"]
    return model


def load_checkpoint(model, checkpoint_name):
    assert os.path.isfile(checkpoint_name) and checkpoint_name.endswith(
        ".tar"
    ), "The specified checkpoint_name is not a valid checkpoint"
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint_name}")
    return model


def check_onnx(onnx_name):
    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["UNet", "UNetDNP", "ConvTasNet", "TransUNet", "SepFormer"])
    ap.add_argument("--checkpoint_name", required=True, help="File with .tar extension")
    ap.add_argument("--onnx_name", default="model.onnx", help="Name of the resulting ONNX model")
    args = ap.parse_args()

    # Load the model in torch format
    model = load_architecture(args.model)
    model = load_checkpoint(model, args.checkpoint_name)

    x = torch.randn(1, 1, 4*16000, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(
        model,                      # model being run
        x,                          # model input (or a tuple for multiple inputs)
        args.onnx_name,             # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=["input"],      # the model's input names
        output_names=["output"],    # the model's output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )

    # Load model again to check the export
    check_onnx(args.onnx_name)


    ort_session = onnxruntime.InferenceSession(args.onnx_name)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")