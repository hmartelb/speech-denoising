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


class Trainer():
    def __init__(self, train_data, val_data, display_freq=50):
        self.train_data = train_data
        self.val_data = val_data

        self.display_freq = display_freq

    def fit(self, model, device, epochs=10, batch_size=16, lr=0.001, weight_decay=1e-5, optimizer=optim.Adam, loss_fn=F.mse_loss, gradient_clipping=True):
        # Get the device placement and make data loaders
        self.device = device
        # model = model.to(self.device)

        kwargs = {'num_workers': 1,
                  'pin_memory': True} if device == 'cuda' else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=batch_size, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=batch_size, **kwargs)

        # Training classes
        self.optimizer = optimizer(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn
        self.gradient_clipping = gradient_clipping

        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs+1):
            print(f"\nEpoch {epoch}/{epochs}:")
            train_loss = self.train(model)
            test_loss = self.test(model)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
        return self.history

    def train(self, model):
        total_loss = 0.0
        model.train()
        with tqdm(self.train_loader) as progress:
            for i, (input_data, output_data) in enumerate(progress):
                # if self.device == 'cuda':
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()

                predictions = model(input_data)
                loss = self.loss_fn(predictions, output_data)

                loss.backward()

                # Gradient Value Clipping
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), clip_value=1.0)

                self.optimizer.step()

                total_loss += loss.item()

                if i % self.display_freq == 0:
                    progress.set_postfix({
                        'loss': float(total_loss/(i+1)),
                    })

        total_loss /= len(self.train_loader)
        return total_loss

    def test(self, model):
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(self.val_loader) as progress:
                for i, (data, label) in enumerate(progress):
                    # if self.device == 'cuda':
                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)

                    predictions = model(input_data)

                    loss = self.loss_fn(predictions, output_data)

                    total_loss += loss.item()

                    if i % self.display_freq == 0:
                        progress.set_postfix({
                            'loss': float(total_loss/(i+1)),
                        })

        total_loss /= len(self.val_loader)
        return total_loss


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Datasets
    ap.add_argument('--clean_dataset_path', required=True)
    ap.add_argument('--noise_dataset_path', required=True)
    ap.add_argument('--target_sr', default=16000, type=int)
    ap.add_argument('--length_seconds', default=4, type=int)

    # Training params
    ap.add_argument('--epochs', default=10, type=int)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--lr', default=1e-4, type=float)
    ap.add_argument('--gradient_clipping', action='store_true')

    # GPU setup
    ap.add_argument('--gpu', default='-1')

    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    visible_devices = list(map(lambda x: int(x), args.gpu.split(',')))
    print("Visible devices:", visible_devices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    from torchaudio.models import ConvTasNet
    model = ConvTasNet(
        num_sources=2,                  # DO NOT CHANGE THIS!
        enc_kernel_size=16,             # 16
        enc_num_feats=128,              # 512
        msk_kernel_size=3,              # 3
        msk_num_feats=32,               # 128
        msk_num_hidden_feats=128,       # 512
        msk_num_layers=8,               # 8
        msk_num_stacks=3                # 3
    )
    model = model.to(device)

    # from models import UNet

    # model = UNet(1, 2)
    # model = torch.nn.DataParallel(
    #     model, device_ids=list(range(len(visible_devices))))

    from data import AudioDirectoryDataset, NoiseMixerDataset

    train_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_dataset_path),
        noise_dataset=AudioDirectoryDataset(root=args.noise_dataset_path),
        mode='time'
    )

    # FIXME: This is the training data repeated, change to validation dirs
    val_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_dataset_path),
        noise_dataset=AudioDirectoryDataset(root=args.noise_dataset_path),
        mode='time'
    )

    # mixture, sources = train_data[0]
    # print("Mixture:", mixture.size(), mixture.min(), mixture.max(), mixture.mean())

    # trainer = Trainer(train_data, val_data)
    # history = trainer.fit(
    #     model,
    #     device,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     loss_fn=F.mse_loss,
    #     gradient_clipping=args.gradient_clipping
    # )
    # exit()

    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()

    kwargs = {'num_workers': 1,
              'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, **kwargs)

    # Training classes
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    gradient_clipping = args.gradient_clipping

    total_loss = 0.0
    display_freq = 10

    torch.autograd.set_detect_anomaly(True)

    model.train()
    with tqdm(train_loader) as progress:
        for i, (input_data, output_data) in enumerate(progress):
            # if device == 'cuda':
            input_data = input_data.cuda()
            output_data = output_data.cuda()

            optimizer.zero_grad()

            predictions = model(input_data)
            loss = loss_fn(predictions, output_data)

            loss.backward()

            # Gradient Value Clipping
            if gradient_clipping:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=1.0)

            optimizer.step()

            total_loss += loss.item()

            if i % display_freq == 0:
                progress.set_postfix({
                    'loss': loss.item(),
                    # 'loss': float(total_loss/(i+1)),
                })

    total_loss /= len(train_loader)
