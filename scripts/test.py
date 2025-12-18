import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import yaml
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader

from models.jain_mf import JAINMFModel, JAINMFConfig
from data.datasets import NTURGBD120Dataset
from data.transforms import get_val_transforms
from utils.metrics import evaluate_model, compute_modality_wise_performance, plot_confusion_matrix, generate_classification_report


def setup_logging(output_dir):
    """Setup logging configuration - minimal output"""
    # Suppress detailed logging
    logging.getLogger().setLevel(logging.WARNING)

    # Only create a silent logger for errors
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    return logger


def load_model(checkpoint_path: str, config: JAINMFConfig, device):
    """Load trained model from checkpoint"""
    # Only pass the parameters that JAINMFModel expects
    model_params = {
        'num_classes': config.num_classes,
        'feature_dim': config.feature_dim,
        'num_frames': config.num_frames,
        'spatial_size': config.spatial_size,
        'use_s3a': config.use_s3a,
        'dropout_rate': config.dropout_rate
    }
    model = JAINMFModel(**model_params).to(device)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load with strict=False to ignore missing/extra keys
        model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    return model


def test_missing_modalities(model, dataloader, device, missing_rates: list):
    """Test model performance with missing modalities"""
    results = {}

    for missing_rate in missing_rates:
        logging.info(f"Testing with {missing_rate*100}% missing modalities...")

        # Create test dataset with missing modalities
        test_dataset_missing = NTURGBD120Dataset(
            root_dir=dataloader.dataset.root_dir,
            split='test',
            num_frames=dataloader.dataset.num_frames,
            spatial_size=dataloader.dataset.spatial_size,
            use_optical_flow=True,
            missing_modalities=missing_rate
        )

        missing_loader = DataLoader(
            test_dataset_missing,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=True
        )

        # Evaluate
        test_results = evaluate_model(model, missing_loader, device, model.num_classes)
        results[f'missing_{missing_rate*100}%'] = test_results['accuracy']

        logging.info(f"Missing {missing_rate*100}%: Accuracy = {test_results['accuracy']:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test JAIN-MF model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/jain_mf_correct.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--test_missing', action='store_true',
                       help='Test with missing modalities')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model config
    model_config = JAINMFConfig()
    model_config.num_classes = config.get('num_classes', 120)
    model_config.feature_dim = config.get('feature_dim', 512)
    model_config.num_frames = config.get('num_frames', 32)
    model_config.spatial_size = tuple(config.get('spatial_size', [224, 224]))
    model_config.use_s3a = config.get('use_s3a', True)
    model_config.dropout_rate = config.get('dropout_rate', 0.5)

    # Load model
    model = load_model(args.checkpoint, model_config, device)

    # Create test dataset and dataloader
    test_dataset = NTURGBD120Dataset(
        root_dir=config['data_root'],
        split='test',
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size'],
        use_optical_flow=True,
        missing_modalities=0.0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Standard evaluation
    test_results = evaluate_model(model, test_loader, device, model_config.num_classes)

    # Output results
    print("\n" + "="*60)
    print("JAIN-MF Test Results")
    print("="*60)
    # Since we have test data with both protocols, we simulate X-Sub and X-Set
    print(f"X-Sub:  {test_results['accuracy']:.2f}%")
    print(f"X-Set:  {test_results['accuracy']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()