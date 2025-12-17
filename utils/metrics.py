import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)

    Returns:
        accuracy: Classification accuracy
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def compute_top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """
    Compute top-k classification accuracy

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        k: Value of k for top-k accuracy

    Returns:
        top_k_accuracy: Top-k classification accuracy
    """
    batch_size = labels.size(0)
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return 100.0 * correct_k / batch_size


def compute_class_wise_accuracy(outputs: torch.Tensor, labels: torch.Tensor,
                               num_classes: int) -> Dict[int, float]:
    """
    Compute class-wise accuracy

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        num_classes: Number of classes

    Returns:
        class_accuracies: Dictionary mapping class indices to accuracies
    """
    _, predicted = outputs.max(1)

    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for i in range(len(labels)):
        label = labels[i]
        class_total[label] += 1
        if predicted[i] == label:
            class_correct[label] += 1

    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    return class_accuracies


def compute_confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor,
                           num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        num_classes: Number of classes

    Returns:
        confusion_mat: Confusion matrix of shape (num_classes, num_classes)
    """
    _, predicted = outputs.max(1)
    confusion_mat = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(),
                                   labels=list(range(num_classes)))
    return confusion_mat


def compute_precision_recall_f1(outputs: torch.Tensor, labels: torch.Tensor,
                               average: str = 'macro') -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    _, predicted = outputs.max(1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu().numpy(),
        predicted.cpu().numpy(),
        average=average,
        zero_division=0
    )

    return precision, recall, f1


def compute_mAP(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute mean Average Precision (mAP)

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)

    Returns:
        map_score: Mean Average Precision score
    """
    num_classes = outputs.shape[1]
    ap_scores = []

    for class_idx in range(num_classes):
        # Binary relevance for this class
        class_labels = (labels == class_idx).float()
        class_scores = torch.softmax(outputs, dim=1)[:, class_idx]

        # Sort by confidence
        sorted_indices = torch.argsort(class_scores, descending=True)
        sorted_labels = class_labels[sorted_indices]

        # Compute AP
        precision_at_k = []
        num_correct = 0

        for k in range(len(sorted_labels)):
            if sorted_labels[k] == 1:
                num_correct += 1
                precision_at_k.append(num_correct / (k + 1))

        if len(precision_at_k) > 0:
            ap = sum(precision_at_k) / sum(sorted_labels)
            ap_scores.append(ap)

    return 100.0 * np.mean(ap_scores) if ap_scores else 0.0


def evaluate_model(model, dataloader, device, num_classes: int) -> Dict:
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        num_classes: Number of classes

    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for sample in dataloader:
            rgb_frames = sample['rgb'].to(device)
            flow_frames = sample['flow'].to(device)
            skeleton_data = sample['skeleton'].to(device)
            labels = sample['label'].to(device)

            outputs = model(rgb_frames, flow_frames, skeleton_data, return_attention=False)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    results = {}

    # Basic accuracy metrics
    results['accuracy'] = compute_accuracy(all_outputs, all_labels)
    results['top5_accuracy'] = compute_top_k_accuracy(all_outputs, all_labels, k=5)

    # Class-wise metrics
    results['class_accuracies'] = compute_class_wise_accuracy(all_outputs, all_labels, num_classes)

    # Precision, Recall, F1
    precision, recall, f1 = compute_precision_recall_f1(all_outputs, all_labels)
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1

    # mAP
    results['map'] = compute_mAP(all_outputs, all_labels)

    # Confusion matrix
    results['confusion_matrix'] = compute_confusion_matrix(all_outputs, all_labels, num_classes)

    return results


def plot_confusion_matrix(confusion_mat: np.ndarray, class_names: List[str] = None,
                         save_path: str = None):
    """
    Plot confusion matrix

    Args:
        confusion_mat: Confusion matrix
        class_names: Names of classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_mat))]

    # Normalize confusion matrix
    confusion_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(confusion_mat_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_class_distribution(labels: torch.Tensor, class_names: List[str] = None,
                           save_path: str = None):
    """
    Plot class distribution

    Args:
        labels: Class labels
        class_names: Names of classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 6))

    # Count samples per class
    unique_labels, counts = torch.unique(labels, return_counts=True)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(max(unique_labels) + 1)]

    # Create bar plot
    plt.bar(range(len(unique_labels)), counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(unique_labels)), [class_names[i] for i in unique_labels], rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_modality_wise_performance(model, dataloader, device) -> Dict:
    """
    Compute performance for different modality combinations

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on

    Returns:
        modality_results: Dictionary of results for each modality combination
    """
    model.eval()
    modality_results = {}

    # Define modality scenarios
    scenarios = {
        'rgb_only': {'rgb': True, 'flow': False, 'skeleton': False},
        'skeleton_only': {'rgb': False, 'flow': False, 'skeleton': True},
        'rgb_skeleton': {'rgb': True, 'flow': False, 'skeleton': True},
        'all_modalities': {'rgb': True, 'flow': True, 'skeleton': True}
    }

    for scenario_name, modalities in scenarios.items():
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for sample in dataloader:
                rgb_frames = sample['rgb'].to(device)
                flow_frames = sample['flow'].to(device)
                skeleton_data = sample['skeleton'].to(device)
                labels = sample['label'].to(device)

                # Zero out missing modalities
                if not modalities['rgb']:
                    rgb_frames = torch.zeros_like(rgb_frames)
                if not modalities['flow']:
                    flow_frames = torch.zeros_like(flow_frames)
                if not modalities['skeleton']:
                    skeleton_data = torch.zeros_like(skeleton_data)

                outputs = model(rgb_frames, flow_frames, skeleton_data, return_attention=False)

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Concatenate and compute accuracy
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        accuracy = compute_accuracy(all_outputs, all_labels)
        modality_results[scenario_name] = accuracy

    return modality_results


def generate_classification_report(outputs: torch.Tensor, labels: torch.Tensor,
                                 class_names: List[str] = None) -> str:
    """
    Generate detailed classification report

    Args:
        outputs: Model outputs of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        class_names: Names of classes

    Returns:
        report: Classification report string
    """
    _, predicted = outputs.max(1)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(outputs.shape[1])]

    report = classification_report(
        labels.cpu().numpy(),
        predicted.cpu().numpy(),
        target_names=class_names,
        digits=4
    )

    return report