"""
Training Engine for CT Jaw Defect Segmentation
Optimized for NVIDIA RTX A6000 with mixed precision training
"""

import os
import torch
from tqdm import tqdm
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


def set_deterministic(seed=42):
    """Set seeds for reproducible training."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            import math
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


class Trainer:
    """
    Training engine with A6000-optimized hyperparameters.

    Features:
    - Mixed precision training (AMP)
    - Sliding window validation
    - Automatic checkpointing
    - Progress tracking
    - Learning rate scheduling with warmup
    - Early stopping
    - Gradient clipping
    - Class-weighted loss
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

        # Set deterministic training if seed provided
        if config.get("seed"):
            set_deterministic(config["seed"])

        os.makedirs(output_dir, exist_ok=True)

        # Loss function with optional class weights for imbalanced data
        # Tumor class is often much smaller than bone - weight it higher
        class_weights = config.get("class_weights")
        if class_weights:
            class_weights = torch.tensor(class_weights, device=self.device)

        loss_type = config.get("loss_type", "dice_ce")
        if loss_type == "dice_focal":
            # DiceFocalLoss is better for small structures like tumors
            self.loss_function = DiceFocalLoss(
                to_onehot_y=True,
                softmax=True,
                focal_weight=class_weights,
                lambda_focal=config.get("focal_weight", 1.0)
            )
        else:
            self.loss_function = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                ce_weight=class_weights
            )

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

        # Gradient clipping value
        self.grad_clip = config.get("grad_clip", 1.0)

        # Metric: Dice score (exclude background)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

        # Tracking
        self.best_metric = -1
        self.best_epoch = -1
        self.start_epoch = 0

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_metric = checkpoint.get('best_metric', -1)
            self.best_epoch = checkpoint.get('best_epoch', -1)
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"✓ Resumed from epoch {self.start_epoch} (best dice: {self.best_metric:.4f})")
            return True
        return False

    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Always save last checkpoint for resuming
        last_path = os.path.join(self.output_dir, "last_checkpoint.pth")
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_path)

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
            num_workers=min(num_workers, 4),
            collate_fn=list_data_collate,
            pin_memory=self.device.type == "cuda"
        )

        max_epochs = self.config.get("max_epochs", 300)
        val_interval = self.config.get("val_interval", 5)

        # Learning rate scheduler with warmup
        warmup_epochs = self.config.get("warmup_epochs", 10)
        scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=self.config.get("min_lr", 1e-6)
        )

        # Early stopping
        early_stopping = None
        if self.config.get("early_stopping_patience"):
            early_stopping = EarlyStopping(
                patience=self.config["early_stopping_patience"],
                min_delta=self.config.get("early_stopping_delta", 0.001)
            )

        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Epochs: {max_epochs} (starting from {self.start_epoch})")
        print(f"  Batch Size: {self.config.get('batch_size', 4)}")
        print(f"  Learning Rate: {self.config.get('lr', 1e-4)} (warmup: {warmup_epochs} epochs)")
        print(f"  ROI Size: {self.config.get('roi_size', (128, 128, 128))}")
        print(f"  Training Samples: {len(train_dataset)}")
        print(f"  Validation Samples: {len(val_dataset)}")
        print(f"  Gradient Clipping: {self.grad_clip}")
        print(f"  Loss Type: {self.config.get('loss_type', 'dice_ce')}")
        if early_stopping:
            print(f"  Early Stopping: patience={self.config['early_stopping_patience']}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, max_epochs):
            # Update learning rate
            current_lr = scheduler.step(epoch)

            # Training phase
            self.model.train()
            epoch_loss = 0
            step = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} (lr={current_lr:.2e})")
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

                    # Gradient clipping (unscale first)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward/backward pass (CPU)
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                epoch_loss += loss.item()
                step += 1
                progress.set_postfix({"loss": f"{epoch_loss/step:.4f}"})

            avg_loss = epoch_loss / step
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            # Validation phase
            if (epoch + 1) % val_interval == 0:
                val_dice, per_class_dice = self._validate(val_loader)
                print(f"Validation Dice: {val_dice:.4f} (Bone: {per_class_dice[0]:.4f}, Tumor: {per_class_dice[1]:.4f})")

                # Save checkpoint
                is_best = val_dice > self.best_metric
                if is_best:
                    self.best_metric = val_dice
                    self.best_epoch = epoch + 1
                    print(f"✓ New Best Model Saved (Dice: {val_dice:.4f})")

                self.save_checkpoint(epoch, is_best=is_best)

                # Early stopping check
                if early_stopping and early_stopping(val_dice):
                    print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                    break

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
            Tuple of (mean_dice, per_class_dice)
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

                # Compute metrics
                self.dice_metric(y_pred=val_outputs, y=val_labels)
                self.dice_metric_batch(y_pred=val_outputs, y=val_labels)

        # Aggregate and reset
        mean_dice = self.dice_metric.aggregate().item()
        per_class_dice = self.dice_metric_batch.aggregate().cpu().numpy()

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return mean_dice, per_class_dice
