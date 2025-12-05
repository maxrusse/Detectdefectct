"""
Training Engine for CT Jaw Defect Segmentation
Optimized for NVIDIA RTX A6000 with mixed precision training
"""

import os
import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


class Trainer:
    """
    Training engine with A6000-optimized hyperparameters.

    Features:
    - Mixed precision training (AMP)
    - Sliding window validation
    - Automatic checkpointing
    - Progress tracking
    """

    def __init__(self, model, config, output_dir="./results"):
        """
        Initialize trainer.

        Args:
            model: PyTorch segmentation model
            config: Configuration dictionary with keys:
                - batch_size, lr, max_epochs, val_interval
                - roi_size, workers
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        # Loss function: Dice + Cross-Entropy
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        # Mixed precision support (only for CUDA)
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Metric: Dice score (exclude background)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        # Tracking
        self.best_metric = -1
        self.best_epoch = -1

    def train(self, train_dataset, val_dataset):
        """
        Run training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Path to best model checkpoint
        """
        # Data loaders with optimized settings
        num_workers = self.config.get("workers", 8)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(num_workers, 4),  # Use fewer workers for validation
            collate_fn=list_data_collate,
            pin_memory=self.device.type == "cuda"
        )

        max_epochs = self.config.get("max_epochs", 300)
        val_interval = self.config.get("val_interval", 5)

        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Epochs: {max_epochs}")
        print(f"  Batch Size: {self.config.get('batch_size', 4)}")
        print(f"  Learning Rate: {self.config.get('lr', 1e-4)}")
        print(f"  ROI Size: {self.config.get('roi_size', (128, 128, 128))}")
        print(f"  Training Samples: {len(train_dataset)}")
        print(f"  Validation Samples: {len(val_dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            step = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for batch in progress:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                if self.use_amp:
                    # Mixed precision forward pass (CUDA only)
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, labels)

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward/backward pass (CPU)
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                step += 1
                progress.set_postfix({"loss": f"{epoch_loss/step:.4f}"})

            avg_loss = epoch_loss / step
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            # Validation phase
            if (epoch + 1) % val_interval == 0:
                val_dice = self._validate(val_loader)
                print(f"Validation Dice: {val_dice:.4f}")

                # Save best model
                if val_dice > self.best_metric:
                    self.best_metric = val_dice
                    self.best_epoch = epoch + 1
                    best_path = os.path.join(self.output_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"✓ New Best Model Saved (Dice: {val_dice:.4f})")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best Validation Dice: {self.best_metric:.4f}")
        print(f"  Best Epoch: {self.best_epoch}")
        print(f"{'='*60}\n")

        return os.path.join(self.output_dir, "best_model.pth")

    def _validate(self, val_loader):
        """
        Run validation with sliding window inference.

        Args:
            val_loader: Validation data loader

        Returns:
            Mean Dice score
        """
        self.model.eval()
        roi_size = self.config.get("roi_size", (128, 128, 128))

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)

                # Sliding window inference with overlap
                sw_batch_size = self.config.get("sw_batch_size", 4)
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.get("sw_overlap", 0.5)
                )

                # Post-processing
                n_classes = self.config.get("n_classes", 3)
                val_outputs = [
                    AsDiscrete(argmax=True, to_onehot=n_classes)(i)
                    for i in decollate_batch(val_outputs)
                ]
                val_labels = [
                    AsDiscrete(to_onehot=n_classes)(i)
                    for i in decollate_batch(val_labels)
                ]

                # Compute metric
                self.dice_metric(y_pred=val_outputs, y=val_labels)

        # Aggregate and reset
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        return mean_dice
