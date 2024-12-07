import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

class IntervalEvaluator:
    """Evaluation utilities for interval recognition model."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Trained PyTorch model
            device: Device to evaluate on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a dataset.
        Returns:
            Tuple of (predictions, true_labels)
        """
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.max(1)[1].cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(target.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def evaluate(self, 
                data_loader: DataLoader,
                class_names: List[str]) -> Dict:
        """
        Evaluate model performance.
        Args:
            data_loader: DataLoader containing evaluation data
            class_names: List of class names (interval names)
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, true_labels = self.predict(data_loader)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Get classification report
        report = classification_report(true_labels, predictions,
                                    target_names=class_names,
                                    output_dict=True)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'accuracy': report['accuracy'],
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def plot_confusion_matrix(self,
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(confusion_matrix, 
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()