import torch
import numpy as np
import librosa

class AudioPreprocessor:
    """Preprocessing transformations for audio data."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 fixed_length: int = 256):  # Add fixed length parameter
        """
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: Length of FFT window
            hop_length: Number of samples between successive frames
            fixed_length: Fixed number of time steps for output
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_length = fixed_length
    
    def melspectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Ensure fixed length through padding or cropping
        if mel_spec_db.shape[1] < self.fixed_length:
            # Pad with zeros if too short
            pad_width = self.fixed_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        elif mel_spec_db.shape[1] > self.fixed_length:
            # Crop if too long
            mel_spec_db = mel_spec_db[:, :self.fixed_length]
        
        return torch.from_numpy(mel_spec_db).float()
    
    def mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> torch.Tensor:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Normalize
        mfccs = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-8)
        
        # Ensure fixed length through padding or cropping
        if mfccs.shape[1] < self.fixed_length:
            pad_width = self.fixed_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
        elif mfccs.shape[1] > self.fixed_length:
            mfccs = mfccs[:, :self.fixed_length]
        
        return torch.from_numpy(mfccs).float()