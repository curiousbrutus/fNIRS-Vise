"""
Utilities for model analysis, evaluation, and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, List, Tuple, Optional
import wandb


class ModelEvaluator:
    """Utility class for comprehensive model evaluation and analysis."""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def extract_features(self, dataloader, layer_name: str = "fnirs_encoder") -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract intermediate features from specified layer.
        
        Args:
            dataloader: DataLoader for feature extraction
            layer_name: Name of layer to extract features from
            
        Returns:
            features: Extracted features [N, feature_dim]
            labels: Corresponding labels [N]
        """
        features = []
        labels = []
        
        # Register forward hook
        activation = {}
        def hook_fn(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
            
        # Get the specified layer
        layer = dict(self.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook_fn(layer_name))
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    fnirs = batch["fnirs"].to(self.device)
                    fmri = batch["fmri"].to(self.device)
                    batch_labels = batch["label"].numpy()
                    
                    # Forward pass
                    _ = self.model(fnirs, fmri)
                    
                    # Extract features
                    batch_features = activation[layer_name].cpu().numpy()
                    if batch_features.ndim > 2:
                        batch_features = batch_features.reshape(batch_features.shape[0], -1)
                    
                    features.append(batch_features)
                    labels.append(batch_labels)
                    
        finally:
            handle.remove()
            
        return np.vstack(features), np.concatenate(labels)
    
    def evaluate_classification(self, dataloader) -> Dict:
        """
        Comprehensive classification evaluation.
        
        Returns:
            metrics: Dictionary with accuracy, precision, recall, F1, etc.
        """
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                fnirs = batch["fnirs"].to(self.device)
                fmri = batch["fmri"].to(self.device)
                labels = batch["label"]
                
                logits = self.model(fnirs, fmri)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_logits.extend(F.softmax(logits, dim=-1).cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()
        
        # Classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=[f"Class_{i}" for i in range(4)],
            output_dict=True
        )
        
        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_logits,
            "classification_report": report
        }
    
    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, 
                            class_names: Optional[List[str]] = None, save_path: str = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names or [f"Class {i}" for i in range(cm.shape[1])],
            yticklabels=class_names or [f"Class {i}" for i in range(cm.shape[0])]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_feature_embeddings(self, features: np.ndarray, labels: np.ndarray, 
                              method: str = "tsne", save_path: str = None):
        """
        Plot 2D embeddings of learned features.
        
        Args:
            features: Feature vectors [N, feature_dim]
            labels: Class labels [N]
            method: Dimensionality reduction method ("tsne", "pca")
            save_path: Optional path to save plot
        """
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Reduce dimensions
        embeddings = reducer.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings[:, 0], embeddings[:, 1], 
            c=labels, cmap='viridis', alpha=0.7, s=50
        )
        plt.colorbar(scatter)
        plt.title(f'Feature Embeddings ({method.upper()})')
        plt.xlabel(f'{method.upper()}_1')
        plt.ylabel(f'{method.upper()}_2')
        
        # Add class legend
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter([], [], label=f'Class {label}', c=plt.cm.viridis(i/len(unique_labels)))
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def analyze_attention_weights(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Extract and analyze attention weights (if using attention-based fusion).
        
        Args:
            batch: Single batch of data
            
        Returns:
            attention_weights: Attention weights [B, seq_len, seq_len] or similar
        """
        if not hasattr(self.model.fusion, 'attention'):
            print("Model does not use attention mechanism")
            return None
            
        # Register hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # This depends on your specific attention implementation
            # Modify based on actual attention module structure
            if hasattr(module, 'attn_weights'):
                attention_weights.append(module.attn_weights.detach().cpu().numpy())
        
        handle = self.model.fusion.register_forward_hook(attention_hook)
        
        try:
            with torch.no_grad():
                fnirs = batch["fnirs"].to(self.device)
                fmri = batch["fmri"].to(self.device)
                _ = self.model(fnirs, fmri)
        finally:
            handle.remove()
            
        return np.array(attention_weights) if attention_weights else None


def memory_usage_analysis(model, input_shapes: Dict[str, Tuple]):
    """
    Analyze GPU memory usage for the model.
    
    Args:
        model: PyTorch model
        input_shapes: Dictionary with input tensor shapes
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory analysis")
        return
        
    device = next(model.parameters()).device
    
    # Create dummy inputs
    dummy_inputs = {}
    for name, shape in input_shapes.items():
        dummy_inputs[name] = torch.randn(shape, device=device, requires_grad=True)
    
    # Memory before forward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Forward pass
    if "fnirs" in dummy_inputs and "fmri" in dummy_inputs:
        output = model(dummy_inputs["fnirs"], dummy_inputs["fmri"])
    else:
        output = model(**dummy_inputs)
    
    mem_after_forward = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    mem_after_backward = torch.cuda.memory_allocated() / 1024**2
    peak_mem_final = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Memory Analysis:")
    print(f"  Before: {mem_before:.2f} MB")
    print(f"  After forward: {mem_after_forward:.2f} MB")
    print(f"  After backward: {mem_after_backward:.2f} MB")
    print(f"  Peak memory: {peak_mem_final:.2f} MB")
    print(f"  Forward memory: {mem_after_forward - mem_before:.2f} MB")
    print(f"  Backward memory: {mem_after_backward - mem_after_forward:.2f} MB")
    
    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Parameter memory: {param_memory:.2f} MB")
    
    return {
        "forward_memory": mem_after_forward - mem_before,
        "backward_memory": mem_after_backward - mem_after_forward,
        "peak_memory": peak_mem_final,
        "parameter_memory": param_memory
    }


def compare_models(results_df: pd.DataFrame, metric: str = "val_acc"):
    """
    Compare different model configurations from sweep results.
    
    Args:
        results_df: DataFrame with sweep results
        metric: Metric to compare (should be column in df)
    """
    # Group by transfer mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Model Comparison ({metric})')
    
    # 1. Transfer mode comparison
    ax = axes[0, 0]
    transfer_comparison = results_df.groupby('transfer_mode')[metric].agg(['mean', 'std'])
    transfer_comparison['mean'].plot(kind='bar', yerr=transfer_comparison['std'], ax=ax)
    ax.set_title('By Transfer Mode')
    ax.set_ylabel(metric)
    
    # 2. Distillation alpha (for distill mode only)
    ax = axes[0, 1]
    distill_data = results_df[results_df['transfer_mode'] == 'distill']
    if len(distill_data) > 0:
        alpha_comparison = distill_data.groupby('distill_alpha')[metric].agg(['mean', 'std'])
        alpha_comparison['mean'].plot(kind='bar', yerr=alpha_comparison['std'], ax=ax)
        ax.set_title('Distillation Alpha (Distill Mode)')
        ax.set_ylabel(metric)
    
    # 3. Freeze fMRI comparison
    ax = axes[1, 0]
    freeze_comparison = results_df.groupby('freeze_fmri')[metric].agg(['mean', 'std'])
    freeze_comparison['mean'].plot(kind='bar', yerr=freeze_comparison['std'], ax=ax)
    ax.set_title('Freeze fMRI Adapter')
    ax.set_ylabel(metric)
    
    # 4. Learning rate comparison  
    ax = axes[1, 1]
    lr_comparison = results_df.groupby('learning_rate')[metric].agg(['mean', 'std'])
    lr_comparison['mean'].plot(kind='bar', yerr=lr_comparison['std'], ax=ax)
    ax.set_title('Learning Rate')
    ax.set_ylabel(metric)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    from src.model import FmriGuidedFnirsNet
    from src.data_module import FmriFnirsDataModule
    
    # Create dummy model and data
    model = FmriGuidedFnirsNet()
    dm = FmriFnirsDataModule(batch_size=4)
    
    # Memory analysis
    memory_usage_analysis(model, {
        "fnirs": (4, 52, 200),
        "fmri": (4, 768)
    })
    
    print("Evaluation utilities loaded successfully!")
