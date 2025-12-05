"""
Inference Engine for CT Jaw Defect Segmentation
Supports sliding window inference and batch prediction
"""

import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


class Predictor:
    """
    Inference engine for segmentation prediction and evaluation.

    Features:
    - Sliding window inference
    - Batch processing
    - Quantitative evaluation
    - NIfTI export
    """

    def __init__(self, model, config, device=None):
        """
        Initialize predictor.

        Args:
            model: Trained PyTorch model
            config: Configuration dictionary with 'roi_size'
            device: Target device (default: auto-detect CUDA)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    def predict(self, input_data, return_prob=False):
        """
        Predict segmentation for a single input.

        Args:
            input_data: Tensor of shape (1, 1, H, W, D)
            return_prob: If True, return probabilities instead of labels

        Returns:
            Segmentation prediction (numpy array)
        """
        roi_size = self.config.get("roi_size", (128, 128, 128))

        with torch.no_grad():
            input_tensor = input_data.to(self.device)

            # Sliding window inference
            sw_batch_size = self.config.get("sw_batch_size", 4)
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=self.config.get("sw_overlap", 0.5)
            )

            if return_prob:
                # Return softmax probabilities
                return torch.softmax(output, dim=1).cpu().numpy()
            else:
                # Return argmax labels
                return torch.argmax(output, dim=1).cpu().numpy()

    def evaluate(self, test_dataset, verbose=True):
        """
        Evaluate model on test dataset.

        Args:
            test_dataset: Test dataset
            verbose: Print results per sample

        Returns:
            Dictionary with mean Dice scores per class
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=list_data_collate
        )

        print(f"\n{'='*60}")
        print(f"Running Evaluation on {len(test_dataset)} samples...")
        print(f"{'='*60}\n")

        roi_size = self.config.get("roi_size", (128, 128, 128))

        with torch.no_grad():
            for idx, test_data in enumerate(tqdm(test_loader, desc="Testing")):
                test_inputs = test_data["image"].to(self.device)
                test_labels = test_data["label"].to(self.device)

                # Sliding window inference
                sw_batch_size = self.config.get("sw_batch_size", 4)
                test_outputs = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.get("sw_overlap", 0.5)
                )

                # Post-processing
                n_classes = self.config.get("n_classes", 3)
                test_outputs = [
                    AsDiscrete(argmax=True, to_onehot=n_classes)(i)
                    for i in decollate_batch(test_outputs)
                ]
                test_labels = [
                    AsDiscrete(to_onehot=n_classes)(i)
                    for i in decollate_batch(test_labels)
                ]

                # Compute metric
                self.dice_metric(y_pred=test_outputs, y=test_labels)

        # Aggregate results
        metrics = self.dice_metric.aggregate()
        self.dice_metric.reset()

        # Print results
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Bone Dice:  {metrics[0].item():.4f}")
        print(f"  Tumor Dice: {metrics[1].item():.4f}")
        print(f"  Mean Dice:  {metrics.mean().item():.4f}")
        print(f"{'='*60}\n")

        return {
            "bone_dice": metrics[0].item(),
            "tumor_dice": metrics[1].item(),
            "mean_dice": metrics.mean().item()
        }

    def save_prediction(self, prediction, reference_path, output_path):
        """
        Save prediction as NIfTI file.

        Args:
            prediction: Numpy array (H, W, D)
            reference_path: Path to reference NIfTI for header info
            output_path: Output file path
        """
        # Load reference header
        ref_nii = nib.load(reference_path)

        # Create new NIfTI
        pred_nii = nib.Nifti1Image(
            prediction.astype(np.uint8),
            affine=ref_nii.affine,
            header=ref_nii.header
        )

        # Save
        nib.save(pred_nii, output_path)
        print(f"✓ Saved prediction to {output_path}")
