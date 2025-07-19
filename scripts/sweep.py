"""
Hyperparameter sweep script using Hydra for automated ablation studies.

This script launches multiple training runs with different configurations
to find optimal transfer learning settings.
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import subprocess
import sys
from pathlib import Path
import itertools


@hydra.main(config_path="../configs", config_name="sweep", version_base=None)
def sweep_main(cfg: DictConfig) -> None:
    """Main sweep function using Hydra configuration."""
    
    print("=== fMRI-fNIRS Transfer Learning Sweep ===")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Extract sweep parameters
    transfer_modes = cfg.sweep.transfer_mode
    distill_alphas = cfg.sweep.distill_alpha  
    freeze_fmri_options = cfg.sweep.freeze_fmri
    learning_rates = cfg.sweep.learning_rate
    batch_sizes = cfg.sweep.batch_size
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        transfer_modes, distill_alphas, freeze_fmri_options, 
        learning_rates, batch_sizes
    ))
    
    print(f"Total experiments: {len(param_combinations)}")
    
    results = []
    
    for i, (transfer_mode, alpha, freeze_fmri, lr, bs) in enumerate(param_combinations, 1):
        
        # Skip invalid combinations
        if transfer_mode != "distill" and alpha != 0.5:
            continue  # Only vary alpha for distillation
            
        print(f"\n--- Experiment {i}/{len(param_combinations)} ---")
        print(f"Mode: {transfer_mode}, Alpha: {alpha}, Freeze: {freeze_fmri}, LR: {lr}, BS: {bs}")
        
        # Build command
        cmd = [
            sys.executable, "src/train.py",
            "--transfer_mode", str(transfer_mode),
            "--distill_alpha", str(alpha),
            "--learning_rate", str(lr),
            "--batch_size", str(bs),
            "--max_epochs", str(cfg.training.max_epochs),
            "--seed", str(cfg.training.seed),
            "--experiment_name", f"sweep_{transfer_mode}_a{alpha}_f{freeze_fmri}_lr{lr}_bs{bs}",
            "--tags", "sweep", transfer_mode, f"alpha_{alpha}"
        ]
        
        if freeze_fmri:
            cmd.append("--freeze_fmri")
            
        # Add other fixed parameters
        if cfg.data.fnirs_data_path:
            cmd.extend(["--fnirs_data_path", cfg.data.fnirs_data_path])
        if cfg.data.fmri_data_path:
            cmd.extend(["--fmri_data_path", cfg.data.fmri_data_path])
            
        try:
            # Run experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2h timeout
            
            if result.returncode == 0:
                print(f"✓ Experiment completed successfully")
                results.append({
                    "transfer_mode": transfer_mode,
                    "distill_alpha": alpha,
                    "freeze_fmri": freeze_fmri,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "status": "success"
                })
            else:
                print(f"✗ Experiment failed with code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                results.append({
                    "transfer_mode": transfer_mode,
                    "distill_alpha": alpha,
                    "freeze_fmri": freeze_fmri,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "status": "failed",
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"✗ Experiment timed out")
            results.append({
                "transfer_mode": transfer_mode,
                "distill_alpha": alpha,
                "freeze_fmri": freeze_fmri,
                "learning_rate": lr,
                "batch_size": bs,
                "status": "timeout"
            })
            
        except Exception as e:
            print(f"✗ Experiment error: {e}")
            results.append({
                "transfer_mode": transfer_mode,
                "distill_alpha": alpha,
                "freeze_fmri": freeze_fmri,
                "learning_rate": lr,
                "batch_size": bs,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print(f"\n=== Sweep Summary ===")
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] != "success"])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    results_path = Path("sweep_results.csv")
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


def manual_sweep():
    """Manual sweep without Hydra (fallback option)."""
    
    # Define sweep parameters
    configs = {
        "transfer_mode": ["feature_guided", "distill", "weight_init"],
        "distill_alpha": [0.0, 0.3, 0.5, 0.7],
        "freeze_fmri": [True, False],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [4, 8]
    }
    
    # Generate combinations (filtered)
    experiments = []
    
    for mode in configs["transfer_mode"]:
        for freeze in configs["freeze_fmri"]:
            for lr in configs["learning_rate"]:
                for bs in configs["batch_size"]:
                    
                    if mode == "distill":
                        # Test different alpha values for distillation
                        for alpha in configs["distill_alpha"]:
                            experiments.append({
                                "transfer_mode": mode,
                                "distill_alpha": alpha,
                                "freeze_fmri": freeze,
                                "learning_rate": lr,
                                "batch_size": bs
                            })
                    else:
                        # Fixed alpha for non-distillation modes
                        experiments.append({
                            "transfer_mode": mode,
                            "distill_alpha": 0.5,  # Unused
                            "freeze_fmri": freeze,
                            "learning_rate": lr,
                            "batch_size": bs
                        })
    
    print(f"Manual sweep: {len(experiments)} experiments")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n--- Manual Experiment {i}/{len(experiments)} ---")
        
        cmd = [
            sys.executable, "src/train.py",
            "--transfer_mode", exp["transfer_mode"],
            "--distill_alpha", str(exp["distill_alpha"]),
            "--learning_rate", str(exp["learning_rate"]),
            "--batch_size", str(exp["batch_size"]),
            "--max_epochs", "50",  # Reduced for sweep
            "--experiment_name", f"manual_sweep_{i}",
            "--tags", "manual_sweep", exp["transfer_mode"]
        ]
        
        if exp["freeze_fmri"]:
            cmd.append("--freeze_fmri")
            
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, timeout=3600)  # 1h timeout
            print(f"✓ Experiment {i} completed")
        except Exception as e:
            print(f"✗ Experiment {i} failed: {e}")


if __name__ == "__main__":
    # Try Hydra first, fallback to manual if config missing
    try:
        sweep_main()
    except Exception as e:
        print(f"Hydra sweep failed: {e}")
        print("Falling back to manual sweep...")
        manual_sweep()
