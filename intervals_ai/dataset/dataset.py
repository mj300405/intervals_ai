import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import librosa
from ..generation.config import IntervalDefinitions

class IntervalsDataset(Dataset):
    """Dataset class for musical intervals."""
    
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[callable] = None,
                 sample_rate: int = 44100):
        """
        Args:
            data_dir: Directory containing the audio files
            transform: Optional transform to be applied on a sample
            sample_rate: Sample rate for loading audio files
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.intervals = IntervalDefinitions()
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.intervals.INTERVALS.keys())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Get all wav files
        self.files = list(self.data_dir.glob("*.wav"))
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load audio file
        audio_path = str(self.files[idx])
        signal, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract label from filename
        label = self._extract_label_from_filename(self.files[idx].name)
        label_idx = self.label_to_idx[label]
        
        # Apply transform if specified (while signal is still a numpy array)
        if self.transform:
            signal = self.transform(signal)
            
        # Now convert to tensor after the transform
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(signal).float()
            
        return signal, label_idx
    
    def _extract_label_from_filename(self, filename: str) -> str:
        """Extract interval label from filename."""
        # First, remove the .wav extension
        filename = filename.replace('.wav', '')
        
        # Split into parts
        parts = filename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {filename}")
        
        # First part should be "interval"
        if parts[0] != "interval":
            raise ValueError(f"Invalid filename format: {filename}")
        
        # Second part should be a number
        if not parts[1].isdigit():
            raise ValueError(f"Invalid filename format: {filename}")
        
        # The rest of the parts form the label
        label = '_'.join(parts[2:])
        
        if label not in self.intervals.INTERVALS:
            raise ValueError(f"Unknown interval label in filename: {label}")
        
        return label