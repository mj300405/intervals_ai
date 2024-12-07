# Intervals AI

A deep learning project for musical interval recognition using PyTorch. This system generates synthetic musical intervals, processes them into mel spectrograms, and trains a CNN model to recognize different musical intervals.

## Project Structure

```
intervals-ai/
│
├── intervals_ai/
│   ├── dataset/
│   │   ├── dataset.py         # Dataset handling
│   │   └── preprocessor.py    # Audio preprocessing
│   ├── generation/
│   │   ├── config.py          # Configuration classes
│   │   ├── intervals.py       # Interval generation
│   │   └── notes.py          # Note generation utilities
│   ├── models/
│   │   ├── cnn.py            # CNN model architecture
│   │   ├── evaluation.py     # Model evaluation
│   │   └── trainer.py        # Training logic
│   ├── utils/
│   │   ├── audio.py          # Audio utilities
│   │   └── visualization.py  # Visualization utilities
│   └── main.py               # Main training script
│
├── generated_data/           # Generated audio samples
│   └── interval_*.wav        # Generated interval audio files
│
├── experiments/             # Training runs
│   └── run_YYYYMMDD_HHMMSS/ # Timestamped experiment directory
│       ├── checkpoints/     # Model checkpoints
│       ├── plots/           # Generated visualizations
│       └── stats/           # Training statistics and metrics
│
├── README.md
├── pyproject.toml          # Poetry dependencies
└── poetry.lock            # Poetry lock file
```

## Setup

1. Install Python 3.10 or newer
2. Install Poetry (dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/mj300405/intervals_ai.git
   cd intervals-ai
   ```
4. Install dependencies:
   ```bash
   poetry install
   ```

## Running the Project

1. Activate the virtual environment:
   ```bash
   poetry shell
   ```

2. Run the training script:
   ```bash
   python intervals_ai/main.py
   ```

The script will:
1. Generate a dataset of 5000 interval samples (if not already present)
2. Train a CNN model for 100 epochs
3. Save results in a timestamped experiment directory

## Experiment Output Structure

Each training run creates a timestamped directory under `experiments/`:

```
experiments/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pt           # Best performing model
│   └── checkpoint_epoch_*.pt   # Regular checkpoints
│
├── plots/
│   ├── confusion_matrix.png    # Model performance matrix
│   ├── training_history.png    # Loss and accuracy curves
│   └── audio_features_*.png    # Audio visualizations
│
└── stats/
    ├── run_config.json         # Run configuration
    ├── training_history.json   # Detailed training metrics
    ├── evaluation_results.json # Test set results
    ├── metrics_summary.txt     # Performance summary
    └── training_summary.txt    # Training overview
```

## Model Parameters

- Input: Mel spectrograms (128 mel bands)
- Architecture: CNN with 3 convolutional layers
- Training:
  - 5000 total samples
  - 70/15/15 train/val/test split
  - 150 epochs maximum
  - Early stopping with 15 epochs patience
  - Learning rate: 0.001
  - Batch size: 32

## Audio Generation

The system generates intervals using these parameters:
- Sample rate: 44100 Hz
- Duration: 2.0 seconds
- Note gap: 0.2 seconds
- Frequency range: 55 Hz (A1) to 1760 Hz (A6)
- Supported intervals:
  - Minor/Major Second
  - Minor/Major Third
  - Perfect Fourth
  - Tritone
  - Perfect Fifth
  - Minor/Major Sixth
  - Minor/Major Seventh
  - Octave

## Visualization Outputs

1. Training Plots:
   - Loss curves (training and validation)
   - Accuracy curves (training and validation)
   - Confusion matrix

2. Audio Visualizations:
   - Waveform
   - Spectrogram
   - Mel spectrogram

## Dependencies

Main dependencies include:
- PyTorch (deep learning)
- librosa (audio processing)
- numpy (numerical computations)
- matplotlib (visualization)
- soundfile (audio I/O)
- scikit-learn (evaluation metrics)

For a complete list, see `pyproject.toml`.

## License

MIT License

## Important Notes

- First run will generate the dataset, which may take some time
- Subsequent runs will reuse the existing dataset
- Each run creates a new experiment directory
- GPU will be used if available, otherwise CPU
- Training progress is saved regularly, allowing for recovery