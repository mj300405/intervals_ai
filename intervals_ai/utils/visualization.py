import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple
import seaborn as sns

class AudioVisualizer:
    """Utility class for visualizing audio data and spectrograms."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def plot_waveform(self,
                     audio_data: np.ndarray,
                     title: Optional[str] = "Waveform",
                     figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot audio waveform.
        
        Args:
            audio_data: Audio signal to plot
            title: Plot title
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        time = np.arange(len(audio_data)) / self.sample_rate
        plt.plot(time, audio_data)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_spectrogram(self,
                        audio_data: np.ndarray,
                        title: Optional[str] = "Spectrogram",
                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot spectrogram of audio data.
        
        Args:
            audio_data: Audio signal to plot
            title: Plot title
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='log'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_mel_spectrogram(self,
                           audio_data: np.ndarray,
                           n_mels: int = 128,
                           title: Optional[str] = "Mel Spectrogram",
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot mel spectrogram of audio data.
        
        Args:
            audio_data: Audio signal to plot
            n_mels: Number of mel bands
            title: Plot title
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        mel_spect = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=n_mels
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(
            mel_spect_db,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_chromagram(self,
                       audio_data: np.ndarray,
                       title: Optional[str] = "Chromagram",
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot chromagram of audio data.
        
        Args:
            audio_data: Audio signal to plot
            title: Plot title
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        chromagram = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        librosa.display.specshow(
            chromagram,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='chroma'
        )
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_features(self,
                             audio_data: np.ndarray,
                             figsize: Tuple[int, int] = (12, 12)) -> None:
        """
        Plot multiple audio features in subplots.
        
        Args:
            audio_data: Audio signal to plot
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Waveform
        time = np.arange(len(audio_data)) / self.sample_rate
        axes[0].plot(time, audio_data)
        axes[0].set_title("Waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        
        # Spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='log',
            ax=axes[1]
        )
        axes[1].set_title("Spectrogram")
        
        # Mel spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        img = librosa.display.specshow(
            mel_spect_db,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=axes[2]
        )
        axes[2].set_title("Mel Spectrogram")
        
        # Add colorbar
        fig.colorbar(img, ax=axes, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_curves(train_losses: list,
                           val_losses: list,
                           train_accs: Optional[list] = None,
                           val_accs: Optional[list] = None,
                           figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot training and validation curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_accs: Optional list of training accuracies
            val_accs: Optional list of validation accuracies
            figsize: Figure size (width, height)
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot losses
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        
        if train_accs and val_accs:
            # Plot accuracies on secondary y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy')
            ax2.plot(train_accs, label='Training Accuracy', linestyle='--')
            ax2.plot(val_accs, label='Validation Accuracy', linestyle='--')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if train_accs and val_accs:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax1.legend(loc='best')
        
        plt.title('Training Progress')
        plt.tight_layout()
        plt.show()