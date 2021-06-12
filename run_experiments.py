import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.models import ConvTasNet
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from getmodel import get_model
from trainer import Trainer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Datasets
    ap.add_argument("--clean_train_path", required=False, default=os.path.join("datasets", "LibriSpeech_16kHz_4s", "train-clean-100"))
    ap.add_argument("--clean_val_path", required=False, default=os.path.join("datasets", "LibriSpeech_16kHz_4s", "test-clean"))
    ap.add_argument("--noise_train_path", required=False, default=os.path.join("datasets", "UrbanSound8K_16kHz_4s", "train"))
    ap.add_argument("--noise_val_path", required=False, default=os.path.join("datasets", "UrbanSound8K_16kHz_4s", "test"))
    ap.add_argument("--keep_rate", default=1.0, type=float)

    # Training params
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--gradient_clipping", action="store_true")

    # Paths
    ap.add_argument("--checkpoints_folder", required=False, default="checkpoints")
    ap.add_argument("--evaluations_folder", required=False, default=os.path.join('..', 'PROJECT', 'EVALUATION'))
    ap.add_argument("--ground_truth_name", required=False, default="Oracle_mixes_16kHz_4s")

    # GPU setup
    ap.add_argument("--gpu", default="-1")

    args = ap.parse_args()

    assert os.path.isdir(args.checkpoints_folder), "The specified checkpoints folder does not exist"
    assert os.path.isdir(args.evaluations_folder), "The specified evaluations folder does not exist"
    assert os.path.isdir(os.path.join(args.evaluations_folder,args.ground_truth_name)), "The specified ground truth folder does not exist"

    #
    # Set the GPU
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    #
    # Initialize the datasets
    #
    from data import AudioDirectoryDataset, NoiseMixerDataset

    train_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_train_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_train_path, keep_rate=args.keep_rate),
        # mode=data_mode,
    )

    val_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_val_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_val_path, keep_rate=args.keep_rate),
        # mode=data_mode,
    )

    #
    # Experiments
    #
    experiments = [
        {"model": "UNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "TransUNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 4},
        {"model": "UNetDNP", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "ConvTasNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
    ]

    for experiment in experiments:

        # Select the model to be used for training
        training_utils_dict = get_model(experiment["model"])

        model = training_utils_dict["model"]
        loss_fn = training_utils_dict["loss_fn"]
        loss_mode = training_utils_dict["loss_mode"]

        data_mode = training_utils_dict["data_mode"]
        train_data.mode = data_mode
        val_data.mode = data_mode

        loss_name = "sisdr" if data_mode == "time" else "mse"

        model_name = f"{experiment['model']}_{loss_name}_{experiment['lr']}_{experiment['epochs']}_epochs"
        checkpoint_name=os.path.join(args.checkpoints_folder, f"{model_name}.tar")

        print("-"*50)
        print('Model:', experiment['model'])
        print('Checkpoint:', checkpoint_name)
        print('Loss:', loss_name)
        print('')
        print("-"*50)

        # # Start training 
        # model = model.to(device)

        # tr = Trainer(train_data, val_data, checkpoint_name=checkpoint_name)
        # history = tr.fit(
        #     model,
        #     device,
        #     epochs=experiment['epochs'],
        #     batch_size=experiment['batch_size'],
        #     lr=experiment['lr'],
        #     loss_fn=loss_fn,
        #     loss_mode=loss_mode,
        #     gradient_clipping=args.gradient_clipping,
        # )

        # # Predict evaluation data
        # predict_evaluation_data(
        #     evaluation_directory=args.evaluation_path,
        #     output_directory=args.output_path,
        #     model=model,
        #     data_mode=data_mode,
        #     length_seconds=4,
        #     # normalize=True,
        # )