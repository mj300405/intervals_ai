import numpy as np
import librosa
import sounddevice as sd
from typing import Optional, Tuple, List
from pathlib import Path

class AudioUtils:
    """Utility functions for audio processing and manipulation."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio_data, sr
    
    def play_audio(self, audio_data: np.ndarray) -> None:
        """
        Play audio array through speakers.
        
        Args:
            audio_data: Audio signal to play
        """
        sd.play(audio_data, self.sample_rate)
        sd.wait()  # Wait until audio is finished playing
    
    def save_audio(self, 
                  audio_data: np.ndarray, 
                  file_path: str,
                  normalize: bool = True) -> None:
        """
        Save audio array to file.
        
        Args:
            audio_data: Audio signal to save
            file_path: Output file path
            normalize: Whether to normalize audio before saving
        """
        if normalize:
            audio_data = librosa.util.normalize(audio_data)
        librosa.output.write_wav(file_path, audio_data, self.sample_rate)
    
    @staticmethod
    def get_audio_length(audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Get length of audio in seconds.
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Length in seconds
        """
        return len(audio_data) / sample_rate
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio_data: Audio signal to normalize
            
        Returns:
            Normalized audio signal
        """
        return librosa.util.normalize(audio_data)
    
    def trim_silence(self, 
                    audio_data: np.ndarray,
                    threshold_db: float = 20,
                    padding_ms: int = 50) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio_data: Audio signal to trim
            threshold_db: Threshold in decibels below reference to consider as silence
            padding_ms: Milliseconds of padding to keep around sound
            
        Returns:
            Trimmed audio signal
        """
        padding = int((padding_ms / 1000) * self.sample_rate)
        trimmed, _ = librosa.effects.trim(
            audio_data, 
            top_db=threshold_db,
            frame_length=2048,
            hop_length=512
        )
        return librosa.util.fix_length(trimmed, size=len(audio_data))
    
    def time_stretch(self, 
                    audio_data: np.ndarray,
                    rate: float) -> np.ndarray:
        """
        Time stretch audio by a given rate.
        
        Args:
            audio_data: Audio signal to stretch
            rate: Rate to stretch by (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched audio signal
        """
        return librosa.effects.time_stretch(audio_data, rate)
    
    def pitch_shift(self,
                   audio_data: np.ndarray,
                   steps: float) -> np.ndarray:
        """
        Shift pitch of audio by given number of semitones.
        
        Args:
            audio_data: Audio signal to pitch shift
            steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio signal
        """
        return librosa.effects.pitch_shift(
            audio_data,
            sr=self.sample_rate,
            n_steps=steps
        )
    
    def split_audio(self, 
                   audio_data: np.ndarray,
                   segment_length: float) -> List[np.ndarray]:
        """
        Split audio into segments of given length.
        
        Args:
            audio_data: Audio signal to split
            segment_length: Length of each segment in seconds
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * self.sample_rate)
        segments = []
        
        for i in range(0, len(audio_data), segment_samples):
            segment = audio_data[i:i + segment_samples]
            if len(segment) == segment_samples:  # Only keep full-length segments
                segments.append(segment)
        
        return segments