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
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: JAINMFConfig, device):
    """Load trained model from checkpoint"""
    model = JAINMFModel(**config.__dict__).to(device)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from checkpoint: {checkpoint_path}")
        logging.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        logging.info(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
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
    parser.add_argument('--checkpoint', type=str, required=True,
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

    # Setup device and logging
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info(f'Testing on device: {device}')
    logger.info(f'Using checkpoint: {args.checkpoint}')

    # Create model config
    model_config = JAINMFConfig()
    model_config.num_classes = config.get('num_classes', 120)

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

    logger.info(f'Test dataset size: {len(test_dataset)}')

    # Standard evaluation
    logger.info('Running standard evaluation...')
    test_results = evaluate_model(model, test_loader, device, model_config.num_classes)

    # Log results
    logger.info('=== Test Results ===')
    logger.info(f"Accuracy: {test_results['accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {test_results['top5_accuracy']:.2f}%")
    logger.info(f"Precision: {test_results['precision']:.4f}")
    logger.info(f"Recall: {test_results['recall']:.4f}")
    logger.info(f"F1 Score: {test_results['f1_score']:.4f}")
    logger.info(f"mAP: {test_results['map']:.2f}%")

    # Save detailed results
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in test_results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=4)

    # Plot confusion matrix
    confusion_plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_results['confusion_matrix'], save_path=confusion_plot_path)

    # Generate classification report
    all_outputs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for sample in test_loader:
            rgb_frames = sample['rgb'].to(device)
            flow_frames = sample['flow'].to(device)
            skeleton_data = sample['skeleton'].to(device)
            labels = sample['label'].to(device)

            outputs = model(rgb_frames, flow_frames, skeleton_data, return_attention=False)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    report = generate_classification_report(all_outputs, all_labels)
    report_file = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)

    logger.info(f'Results saved to {args.output_dir}')

    # Modality-wise evaluation
    logger.info('Running modality-wise evaluation...')
    modality_results = compute_modality_wise_performance(model, test_loader, device)

    logger.info('=== Modality-wise Results ===')
    for scenario, accuracy in modality_results.items():
        logger.info(f"{scenario}: {accuracy:.2f}%")

    # Save modality results
    modality_file = os.path.join(args.output_dir, 'modality_results.json')
    with open(modality_file, 'w') as f:
        json.dump(modality_results, f, indent=4)

    # Test with missing modalities if requested
    if args.test_missing:
        logger.info('Testing with missing modalities...')
        missing_rates = [0.1, 0.3, 0.5]
        missing_results = test_missing_modalities(model, test_loader, device, missing_rates)

        logger.info('=== Missing Modality Results ===')
        for scenario, accuracy in missing_results.items():
            logger.info(f"{scenario}: {accuracy:.2f}%")

        # Save missing modality results
        missing_file = os.path.join(args.output_dir, 'missing_modality_results.json')
        with open(missing_file, 'w') as f:
            json.dump(missing_results, f, indent=4)

    logger.info('Testing completed successfully!')


if __name__ == '__main__':
    main()