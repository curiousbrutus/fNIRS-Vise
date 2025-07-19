"""
Data preprocessing utilities for fNIRS and fMRI data.

Handles conversion from raw formats to PyTorch tensors,
normalization, artifact removal, and data augmentation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from scipy import signal, io
from scipy.signal import butter, filtfilt, detrend
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt


class FnirsPreprocessor:
    """
    Preprocessor for fNIRS data with standard pipeline:
    1. Motion artifact correction
    2. Bandpass filtering  
    3. Detrending
    4. Channel-wise normalization
    5. Temporal windowing
    """
    
    def __init__(self,
                 sampling_rate: float = 10.0,
                 low_freq: float = 0.01,
                 high_freq: float = 0.2,
                 window_size: int = 200,
                 overlap: float = 0.5,
                 channels: int = 52):
        """
        Initialize preprocessor.
        
        Args:
            sampling_rate: fNIRS sampling rate (Hz)
            low_freq: Lowpass filter cutoff (Hz)
            high_freq: Highpass filter cutoff (Hz) 
            window_size: Time window size (samples)
            overlap: Window overlap ratio (0-1)
            channels: Number of fNIRS channels
        """
        self.fs = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq  
        self.window_size = window_size
        self.overlap = overlap
        self.channels = channels
        
        # Design bandpass filter
        self.sos = self._design_filter()
        
        # Scalers for normalization
        self.channel_scalers = [RobustScaler() for _ in range(channels)]
        self.fitted = False
        
    def _design_filter(self):
        """Design Butterworth bandpass filter."""
        nyquist = self.fs / 2
        low_norm = self.low_freq / nyquist
        high_norm = self.high_freq / nyquist
        
        # 4th order Butterworth bandpass
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        return sos
    
    def _motion_correction(self, data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """
        Simple motion artifact correction using threshold-based detection.
        
        Args:
            data: Raw fNIRS data [channels, time]
            threshold: Motion detection threshold (standard deviations)
            
        Returns:
            corrected: Motion-corrected data
        """
        corrected = data.copy()
        
        for ch in range(data.shape[0]):
            # Detect motion artifacts as outliers
            ch_data = data[ch]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            
            # Find outliers
            outliers = np.abs(ch_data - mean_val) > threshold * std_val
            
            if np.any(outliers):
                # Simple interpolation for artifact correction
                outlier_indices = np.where(outliers)[0]
                for idx in outlier_indices:
                    # Linear interpolation between neighbors
                    if idx > 0 and idx < len(ch_data) - 1:
                        corrected[ch, idx] = (corrected[ch, idx-1] + corrected[ch, idx+1]) / 2
                        
        return corrected
    
    def _temporal_windowing(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create overlapping temporal windows.
        
        Args:
            data: Preprocessed data [channels, time]
            
        Returns:
            windows: List of windowed data arrays
        """
        windows = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        for start in range(0, data.shape[1] - self.window_size + 1, step_size):
            end = start + self.window_size
            window = data[:, start:end]  # [channels, window_size]
            windows.append(window)
            
        return windows
    
    def fit(self, data_list: List[np.ndarray]):
        """
        Fit normalization parameters on training data.
        
        Args:
            data_list: List of fNIRS recordings [channels, time]
        """
        print("Fitting normalization parameters...")
        
        # Collect all channel data for fitting
        all_channel_data = [[] for _ in range(self.channels)]
        
        for data in data_list:
            # Preprocess without normalization
            processed = self._preprocess_single(data, normalize=False)
            
            # Collect data for each channel
            for ch in range(self.channels):
                if ch < processed.shape[0]:
                    all_channel_data[ch].extend(processed[ch].flatten())
        
        # Fit scalers for each channel
        for ch in range(self.channels):
            if len(all_channel_data[ch]) > 0:
                channel_data = np.array(all_channel_data[ch]).reshape(-1, 1)
                self.channel_scalers[ch].fit(channel_data)
        
        self.fitted = True
        print("Normalization parameters fitted.")
    
    def _preprocess_single(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess single fNIRS recording.
        
        Args:
            data: Raw fNIRS data [channels, time]
            normalize: Whether to apply normalization
            
        Returns:
            processed: Preprocessed data [channels, time]
        """
        # 1. Motion artifact correction
        data = self._motion_correction(data)
        
        # 2. Bandpass filtering
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = signal.sosfiltfilt(self.sos, data[ch])
        
        # 3. Detrending
        detrended = np.zeros_like(filtered)
        for ch in range(filtered.shape[0]):
            detrended[ch] = detrend(filtered[ch])
        
        # 4. Channel-wise normalization
        if normalize and self.fitted:
            normalized = np.zeros_like(detrended)
            for ch in range(min(self.channels, detrended.shape[0])):
                ch_data = detrended[ch].reshape(-1, 1)
                normalized[ch] = self.channel_scalers[ch].transform(ch_data).flatten()
            return normalized
        
        return detrended
    
    def preprocess(self, data: np.ndarray) -> List[torch.Tensor]:
        """
        Full preprocessing pipeline.
        
        Args:
            data: Raw fNIRS data [channels, time]
            
        Returns:
            windows: List of preprocessed windows as tensors [channels, window_size]
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Preprocess
        processed = self._preprocess_single(data, normalize=True)
        
        # Create windows
        windows = self._temporal_windowing(processed)
        
        # Convert to tensors
        tensor_windows = [torch.tensor(w, dtype=torch.float32) for w in windows]
        
        return tensor_windows


class FmriPreprocessor:
    """
    Preprocessor for fMRI features (assumes pre-extracted MinD-Vis features).
    
    Handles normalization and dimensionality adjustment for fMRI latent features.
    """
    
    def __init__(self, target_dim: int = 768):
        self.target_dim = target_dim
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, features_list: List[np.ndarray]):
        """Fit normalization on fMRI features."""
        all_features = np.vstack(features_list)
        self.scaler.fit(all_features)
        self.fitted = True
        
    def preprocess(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess fMRI features.
        
        Args:
            features: Raw fMRI features [feature_dim] or [1, feature_dim]
            
        Returns:
            processed: Normalized features [target_dim]
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted.")
            
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Normalize
        normalized = self.scaler.transform(features)
        
        # Adjust dimensionality
        if normalized.shape[1] != self.target_dim:
            if normalized.shape[1] > self.target_dim:
                # Truncate
                normalized = normalized[:, :self.target_dim]
            else:
                # Pad with zeros
                padding = np.zeros((normalized.shape[0], self.target_dim - normalized.shape[1]))
                normalized = np.hstack([normalized, padding])
        
        # Convert to tensor
        return torch.tensor(normalized.squeeze(), dtype=torch.float32)


def load_matlab_data(mat_path: str) -> Dict[str, np.ndarray]:
    """
    Load fNIRS data from MATLAB file.
    
    Args:
        mat_path: Path to .mat file
        
    Returns:
        data_dict: Dictionary with loaded data
    """
    try:
        mat_data = io.loadmat(mat_path)
        
        # Remove MATLAB metadata
        data_dict = {k: v for k, v in mat_data.items() 
                    if not k.startswith('__')}
        
        return data_dict
    except Exception as e:
        print(f"Error loading MATLAB file {mat_path}: {e}")
        return {}


def convert_raw_to_tensors(raw_data_dir: str, 
                          output_dir: str,
                          fnirs_preprocessor: FnirsPreprocessor,
                          fmri_preprocessor: FmriPreprocessor):
    """
    Convert raw data files to preprocessed PyTorch tensors.
    
    Args:
        raw_data_dir: Directory with raw data files
        output_dir: Directory to save processed tensors
        fnirs_preprocessor: Fitted fNIRS preprocessor
        fmri_preprocessor: Fitted fMRI preprocessor
    """
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all data files
    for data_file in raw_path.glob("*.mat"):
        print(f"Processing {data_file.name}...")
        
        try:
            # Load data
            data_dict = load_matlab_data(str(data_file))
            
            # Extract fNIRS data (adjust keys based on your data structure)
            if 'fnirs_data' in data_dict:
                fnirs_raw = data_dict['fnirs_data']  # Should be [channels, time]
            else:
                print(f"No fNIRS data found in {data_file.name}")
                continue
            
            # Extract fMRI features (if available)
            fmri_features = data_dict.get('fmri_features', np.random.randn(768))  # Placeholder
            
            # Preprocess fNIRS
            fnirs_windows = fnirs_preprocessor.preprocess(fnirs_raw)
            
            # Preprocess fMRI
            fmri_tensor = fmri_preprocessor.preprocess(fmri_features)
            
            # Save windowed data
            base_name = data_file.stem
            for i, fnirs_window in enumerate(fnirs_windows):
                # Save fNIRS window
                fnirs_save_path = output_path / f"{base_name}_window_{i:03d}_fnirs.pt"
                torch.save(fnirs_window, fnirs_save_path)
                
                # Save corresponding fMRI features
                fmri_save_path = output_path / f"{base_name}_window_{i:03d}_fmri.pt"
                torch.save(fmri_tensor, fmri_save_path)
                
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")
            
    print(f"Processing complete. Saved to {output_dir}")


def create_data_splits_metadata(data_dir: str, output_path: str):
    """
    Create metadata file for data splits.
    
    Args:
        data_dir: Directory with processed tensor files
        output_path: Path to save metadata CSV
    """
    data_path = Path(data_dir)
    
    samples = []
    for fnirs_file in data_path.glob("*_fnirs.pt"):
        # Parse filename to extract metadata
        base_name = fnirs_file.stem.replace("_fnirs", "")
        parts = base_name.split("_")
        
        # Adjust parsing based on your naming convention
        subject = parts[0] if len(parts) > 0 else "unknown"
        condition = parts[1] if len(parts) > 1 else "unknown"
        window_idx = parts[-1] if parts[-1].isdigit() else "0"
        
        # Find corresponding fMRI file
        fmri_file = data_path / f"{base_name}_fmri.pt"
        
        if fmri_file.exists():
            samples.append({
                'fnirs_path': str(fnirs_file),
                'fmri_path': str(fmri_file),
                'subject': subject,
                'condition': condition,
                'window': int(window_idx),
                'label': 0  # Placeholder - adjust based on your labeling scheme
            })
    
    # Save metadata
    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata for {len(samples)} samples to {output_path}")
    

if __name__ == "__main__":
    # Example preprocessing pipeline
    
    print("=== fNIRS-fMRI Data Preprocessing ===")
    
    # Initialize preprocessors
    fnirs_prep = FnirsPreprocessor(
        sampling_rate=10.0,
        window_size=200,
        channels=52
    )
    
    fmri_prep = FmriPreprocessor(target_dim=768)
    
    # Example: Load and preprocess sample data
    # Adjust paths based on your data structure
    raw_data_dir = "./data/raw"
    processed_data_dir = "./data/processed"
    
    if os.path.exists(raw_data_dir):
        # Load sample data for fitting
        sample_files = list(Path(raw_data_dir).glob("*.mat"))[:5]  # Use first 5 for fitting
        
        if sample_files:
            print("Loading sample data for fitting...")
            fnirs_samples = []
            fmri_samples = []
            
            for mat_file in sample_files:
                data_dict = load_matlab_data(str(mat_file))
                if 'fnirs_data' in data_dict:
                    fnirs_samples.append(data_dict['fnirs_data'])
                if 'fmri_features' in data_dict:
                    fmri_samples.append(data_dict['fmri_features'])
            
            # Fit preprocessors
            if fnirs_samples:
                fnirs_prep.fit(fnirs_samples)
            if fmri_samples:
                fmri_prep.fit(fmri_samples)
            
            # Convert all data
            convert_raw_to_tensors(
                raw_data_dir, processed_data_dir,
                fnirs_prep, fmri_prep
            )
            
            # Create metadata
            create_data_splits_metadata(processed_data_dir, "./data/metadata.csv")
            
        else:
            print("No .mat files found in raw data directory")
    else:
        print(f"Raw data directory {raw_data_dir} not found")
        
    print("Preprocessing pipeline complete!")
