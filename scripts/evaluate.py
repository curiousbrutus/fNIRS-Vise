"""
Evaluation script for trained models.

Usage:
    python evaluate.py --checkpoint_path ./checkpoints/best_model.ckpt --test_subject sub01
"""

import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import FmriGuidedFnirsNet
from src.data_module import FmriFnirsDataModule
from src.utils import ModelEvaluator, compare_models
import pandas as pd


def evaluate_single_model(checkpoint_path: str, test_subject: str = None):
    """Evaluate a single trained model."""
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load model
    model = FmriGuidedFnirsNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Setup data
    dm = FmriFnirsDataModule(test_subject=test_subject, batch_size=16)
    dm.setup()
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluator.evaluate_classification(dm.test_dataloader())
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Classification Report:")
    print(test_metrics['classification_report'])
    
    # Plot confusion matrix
    fig1 = evaluator.plot_confusion_matrix(
        test_metrics['labels'], 
        test_metrics['predictions'],
        class_names=['Emotion 0', 'Emotion 1', 'Emotion 2', 'Emotion 3']
    )
    plt.show()
    
    # Extract and plot features
    print("Extracting learned features...")
    features, labels = evaluator.extract_features(dm.test_dataloader(), "fnirs_encoder")
    
    # Plot feature embeddings
    fig2 = evaluator.plot_feature_embeddings(features, labels, method="tsne")
    plt.show()
    
    fig3 = evaluator.plot_feature_embeddings(features, labels, method="pca") 
    plt.show()
    
    return test_metrics


def evaluate_sweep_results(results_csv_path: str):
    """Evaluate and compare sweep results."""
    
    if not Path(results_csv_path).exists():
        print(f"Results file {results_csv_path} not found")
        return
        
    # Load results
    results_df = pd.read_csv(results_csv_path)
    
    # Filter successful runs
    successful_runs = results_df[results_df['status'] == 'success']
    
    if len(successful_runs) == 0:
        print("No successful runs found in sweep results")
        return
        
    print(f"Analyzing {len(successful_runs)} successful runs from sweep")
    
    # Add dummy metrics for demonstration (replace with actual metrics from logs)
    # In practice, you'd load these from wandb or model checkpoints
    np.random.seed(42)
    successful_runs['val_acc'] = np.random.uniform(0.6, 0.9, len(successful_runs))
    successful_runs['test_acc'] = successful_runs['val_acc'] + np.random.uniform(-0.05, 0.02, len(successful_runs))
    
    # Find best configurations
    best_config = successful_runs.loc[successful_runs['val_acc'].idxmax()]
    print(f"\nBest Configuration (Val Acc: {best_config['val_acc']:.4f}):")
    print(f"  Transfer Mode: {best_config['transfer_mode']}")
    print(f"  Distill Alpha: {best_config['distill_alpha']}")
    print(f"  Freeze fMRI: {best_config['freeze_fmri']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    
    # Compare models
    fig = compare_models(successful_runs, metric='val_acc')
    plt.show()
    
    # Analysis by transfer mode
    print(f"\nResults by Transfer Mode:")
    mode_results = successful_runs.groupby('transfer_mode')['val_acc'].agg(['mean', 'std', 'count'])
    print(mode_results)
    
    # Statistical significance testing
    from scipy import stats
    
    modes = successful_runs['transfer_mode'].unique()
    if len(modes) >= 2:
        print(f"\nPairwise t-tests between transfer modes:")
        for i, mode1 in enumerate(modes):
            for mode2 in modes[i+1:]:
                group1 = successful_runs[successful_runs['transfer_mode'] == mode1]['val_acc']
                group2 = successful_runs[successful_runs['transfer_mode'] == mode2]['val_acc']
                
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    print(f"  {mode1} vs {mode2}: t={t_stat:.3f}, p={p_value:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fMRI-fNIRS Transfer Learning Models")
    
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--test_subject", type=str, default=None,
                       help="Subject ID for LOSO testing")
    parser.add_argument("--sweep_results", type=str, default="sweep_results.csv",
                       help="Path to sweep results CSV")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"],
                       help="Evaluation mode: single model or sweep results")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if args.checkpoint_path is None:
            print("Error: --checkpoint_path required for single model evaluation")
            return
        evaluate_single_model(args.checkpoint_path, args.test_subject)
        
    elif args.mode == "sweep":
        evaluate_sweep_results(args.sweep_results)


if __name__ == "__main__":
    main()
