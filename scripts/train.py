import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
import logging
from datetime import datetime

from models.jain_mf import JAINMFModel, JAINMFConfig
from data.datasets import NTURGBD120Dataset, MixedModalDataset
from data.transforms import get_train_transforms, get_val_transforms, MultiModalAugmentation
from utils.metrics import compute_accuracy, compute_top_k_accuracy
from utils.visualization import log_attention_maps


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, acc, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }

    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))

    # Save epoch checkpoint
    if epoch % 10 == 0:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, args):
    """Train model for one epoch"""
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_s3a_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, sample in enumerate(progress_bar):
        # Move data to device
        rgb_frames = sample['rgb'].to(device)
        flow_frames = sample['flow'].to(device)
        skeleton_data = sample['skeleton'].to(device)
        labels = sample['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(rgb_frames, flow_frames, skeleton_data, return_attention=False)

        # Compute loss
        loss, loss_dict = model.compute_loss(
            outputs, labels, rgb_frames, flow_frames, skeleton_data,
            lambda_ce=args.lambda_ce, lambda_s3a=args.lambda_s3a
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_ce_loss += loss_dict.get('ce_loss', 0)
        if 's3a_reconstruction_loss' in loss_dict:
            total_s3a_loss += loss_dict['s3a_reconstruction_loss']

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

        # Log batch statistics
        if batch_idx % 50 == 0:
            logger.info(f'Epoch {epoch} [{batch_idx}/{len(dataloader)}] '
                       f'Loss: {loss.item():.4f} '
                       f'CE Loss: {loss_dict.get("ce_loss", 0):.4f} '
                       f'Acc: {100.*correct/total:.2f}%')

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_s3a_loss = total_s3a_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    logger.info(f'Train Epoch: {epoch} '
               f'Avg Loss: {avg_loss:.4f} '
               f'Avg CE Loss: {avg_ce_loss:.4f} '
               f'Avg S3A Loss: {avg_s3a_loss:.4f} '
               f'Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, epoch, logger, args):
    """Validate model for one epoch"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc='Validation'):
            # Move data to device
            rgb_frames = sample['rgb'].to(device)
            flow_frames = sample['flow'].to(device)
            skeleton_data = sample['skeleton'].to(device)
            labels = sample['label'].to(device)

            # Forward pass
            outputs = model(rgb_frames, flow_frames, skeleton_data, return_attention=False)

            # Compute loss
            loss, loss_dict = model.compute_loss(
                outputs, labels, rgb_frames, flow_frames, skeleton_data,
                lambda_ce=args.lambda_ce, lambda_s3a=args.lambda_s3a
            )

            # Statistics
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store for top-k accuracy
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    top5_accuracy = compute_top_k_accuracy(all_outputs, all_labels, k=5)

    logger.info(f'Val Epoch: {epoch} '
               f'Loss: {avg_loss:.4f} '
               f'Acc: {accuracy:.2f}% '
               f'Top-5 Acc: {top5_accuracy:.2f}%')

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train JAIN-MF model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    log_dir = os.path.join(config['output_dir'], 'logs')
    logger = setup_logging(log_dir)

    logger.info(f'Starting training on device: {device}')
    logger.info(f'Configuration: {config}')

    # Create model
    model_config = JAINMFConfig()
    model = JAINMFModel(**model_config.__dict__).to(device)

    logger.info(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    # Create datasets and dataloaders
    train_transforms = get_train_transforms(
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size']
    )

    val_transforms = get_val_transforms(
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size']
    )

    # Add multi-modal augmentation
    train_transforms = MultiModalAugmentation(
        missing_modalities=config.get('missing_modalities_prob', {}),
        modality_drop_rate=config.get('modality_drop_rate', 0.1)
    )

    # Create datasets
    train_dataset = NTURGBD120Dataset(
        root_dir=config['data_root'],
        split='train',
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size'],
        use_optical_flow=True,
        missing_modalities=config.get('missing_modalities_prob', 0.0)
    )

    val_dataset = NTURGBD120Dataset(
        root_dir=config['data_root'],
        split='val',
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size'],
        use_optical_flow=True,
        missing_modalities=0.0
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['accuracy']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f'Resumed from checkpoint: {args.resume} (epoch {start_epoch})')
        else:
            logger.warning(f'Checkpoint not found: {args.resume}')

    # Training loop
    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')

    for epoch in range(start_epoch, config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, None, optimizer, device, epoch, logger, config
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, None, device, epoch, logger, config
        )

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc, checkpoint_dir, is_best
        )

    logger.info(f'Training completed. Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()