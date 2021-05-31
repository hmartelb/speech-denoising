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
    def __init__(self, train_data, val_data, checkpoint_name, display_freq=10):
        self.train_data = train_data
        self.val_data = val_data
        assert checkpoint_name.endswith('.tar'), "The checkpoint file must have .tar extension"
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq

    def fit(self, model, device, epochs=10, batch_size=16, lr=0.001, weight_decay=1e-5, optimizer=optim.Adam, loss_fn=F.mse_loss, gradient_clipping=True):
        # Get the device placement and make data loaders
        self.device = device
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, **kwargs)

        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn
        self.gradient_clipping = gradient_clipping
        self.history = {'train_loss': [], 'test_loss': []}
        
        previous_epochs = 0
        min_loss = None

        # Try loading checkpoint (if it exists)
        if os.path.isfile(self.checkpoint_name):
            print(f"Resuming training from checkpoint: {self.checkpoint_name}")
            checkpoint = torch.load(self.checkpoint_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_fn = checkpoint['loss_fn']
            self.history = checkpoint['history']
            previous_epochs = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
        else:
            print(f"No checkpoint found, using default parameters...")
        
        for epoch in range(previous_epochs+1, epochs+1):
            print(f"\nEpoch {epoch}/{epochs}:")
            train_loss = self.train(model)
            test_loss = self.test(model)

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if min_loss is None or test_loss < min_loss:
                print(f"Validation loss improved from {min_loss} to {test_loss}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")
                min_loss = test_loss

                checkpoint_data = {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_fn': self.loss_fn,
                    'history': self.history
                }
                torch.save(checkpoint_data, self.checkpoint_name)

        return self.history

    def train(self, model):
        total_loss = 0.0
        model.train()
        with tqdm(self.train_loader) as progress:
            for i, (input_data, output_data) in enumerate(progress):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()

                predictions = model(input_data)
                loss = self.loss_fn(predictions, output_data)

                loss.backward()

                # Gradient Value Clipping
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

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
                for i, (input_data, output_data) in enumerate(progress):
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
    ap.add_argument('--clean_train_path', required=True)
    ap.add_argument('--clean_val_path', required=True)
    ap.add_argument('--noise_train_path', required=True)
    ap.add_argument('--noise_val_path', required=True)
    ap.add_argument('--keep_rate', default=1.0, type=float)

    # Model checkpoint
    ap.add_argument('--checkpoint_name', required=True, help='File with .tar extension')

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

    # from torchaudio.models import ConvTasNet
    # model = ConvTasNet(
    #     num_sources=2,                  # DO NOT CHANGE THIS!
    #     enc_kernel_size=16,             # 16
    #     enc_num_feats=128,              # 512
    #     msk_kernel_size=3,              # 3
    #     msk_num_feats=32,               # 128
    #     msk_num_hidden_feats=128,       # 512
    #     msk_num_layers=8,               # 8
    #     msk_num_stacks=3                # 3
    # )

    from models import UNet, UNetDNP
    from torchsummary import summary

    # model = UNet(1, 2, unet_scale_factor=16)

    # model = torch.nn.DataParallel(model, device_ids=list(range(len(visible_devices))))
    model = UNetDNP(n_channels=1, n_class=2, unet_depth=6, n_filters=16)
    model = model.to(device)

    # summary(model, input_size=(1, 256, 256))

    from data import AudioDirectoryDataset, NoiseMixerDataset

    train_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_train_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_train_path, keep_rate=args.keep_rate),
        mode='time'#'amplitude'
    )

    val_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_val_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_val_path, keep_rate=args.keep_rate),
        mode='time'#'amplitude'
    )

    trainer = Trainer(train_data, val_data, checkpoint_name=args.checkpoint_name)
    history = trainer.fit(
        model,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_fn=F.mse_loss,
        gradient_clipping=args.gradient_clipping
    )
