import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import librosa
import soundfile as sf
from .config import GenerationConfig, IntervalDefinitions

class IntervalGenerator:
    """Generate musical intervals for training data."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.intervals = IntervalDefinitions()
        self._setup_time_array()
    
    def _setup_time_array(self) -> None:
        """Create time array for signal generation."""
        self.samples = int(self.config.sample_rate * self.config.duration)
        self.t = np.linspace(0, self.config.duration, self.samples)
    
    def generate_note(self, frequency: float) -> np.ndarray:
        """Generate a single note with the given frequency."""
        signal = self.config.amplitude * np.sin(2 * np.pi * frequency * self.t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(signal)
        attack_samples = int(0.01 * self.config.sample_rate)
        decay_samples = int(0.01 * self.config.sample_rate)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return signal * envelope

    def generate_interval(self, base_freq: float, interval_name: str) -> Tuple[np.ndarray, str]:
        """
        Generate an interval starting from base_freq.
        Returns:
            Tuple[np.ndarray, str]: Audio signal and interval label
        """
        if interval_name not in self.intervals.INTERVALS:
            raise ValueError(f"Unknown interval: {interval_name}")
        
        # Calculate the upper note frequency
        semitones = self.intervals.INTERVALS[interval_name]
        upper_freq = base_freq * (2 ** (semitones/12))
        
        # Generate both notes
        note1 = self.generate_note(base_freq)
        note2 = self.generate_note(upper_freq)
        
        # Add gap between notes
        gap_samples = int(self.config.note_gap * self.config.sample_rate)
        gap = np.zeros(gap_samples)
        
        # Combine notes with gap
        combined = np.concatenate([note1, gap, note2])
        
        return combined, interval_name
    
    def generate_dataset(self, 
                        num_samples: int,
                        output_dir: Optional[str] = None
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate a dataset of intervals with random base frequencies.
        Args:
            num_samples: Number of samples to generate
            output_dir: If provided, save audio files to this directory
        """
        dataset = []
        intervals = list(self.intervals.INTERVALS.keys())
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # Random base frequency between min_freq and max_freq
            base_freq = np.random.uniform(self.config.min_freq, self.config.max_freq)
            # Random interval
            interval = np.random.choice(intervals)
            # Generate the interval
            sample, label = self.generate_interval(base_freq, interval)
            dataset.append((sample, label))
            
            # Save audio file if output directory is provided
            if output_dir:
                filename = output_dir / f"interval_{i:04d}_{label}.wav"
                self.save_sample(sample, str(filename))
        
        return dataset

    def save_sample(self, signal: np.ndarray, filename: str) -> None:
        """Save a generated sample to a WAV file."""
        sf.write(filename, signal, self.config.sample_rate)