import torch
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

from intervals_ai.generation.config import GenerationConfig
from intervals_ai.generation.intervals import IntervalGenerator
from intervals_ai.dataset.dataset import IntervalsDataset
from intervals_ai.dataset.preprocessor import AudioPreprocessor
from intervals_ai.models.cnn import IntervalCNN
from intervals_ai.models.trainer import IntervalTrainer
from intervals_ai.models.evaluation import IntervalEvaluator
from intervals_ai.utils.audio import AudioUtils
from intervals_ai.utils.visualization import AudioVisualizer

def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_run_config(config_dir: Path, **kwargs):
    config = {
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    with open(config_dir / 'run_config.json', 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directory structure
    base_dir = Path("experiments")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_dir / f"run_{timestamp}")
    
    data_dir = Path("generated_data")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    plots_dir = ensure_dir(run_dir / "plots")
    stats_dir = ensure_dir(run_dir / "stats")

    # 1. Generate Dataset if it doesn't exist
    if not data_dir.exists() or not list(data_dir.glob("*.wav")):
        print("Generating new dataset...")
        ensure_dir(data_dir)
        config = GenerationConfig(
            sample_rate=44100,
            duration=2.0,
            amplitude=0.5,
            note_gap=0.2,
            min_freq=55.0,
            max_freq=1760.0
        )
        interval_generator = IntervalGenerator(config)
        
        dataset = interval_generator.generate_dataset(
            num_samples=5000,
            output_dir=str(data_dir)
        )
        print(f"Generated {len(dataset)} new samples")
    else:
        print(f"Using existing dataset in {data_dir}")
        interval_generator = IntervalGenerator()  # For intervals definition
    
    # Save run configuration
    save_run_config(
        stats_dir,
        data_directory=str(data_dir),
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        model_params={
            "n_mels": 128,
            "n_intervals": len(interval_generator.intervals.INTERVALS)
        }
    )
    
    # 2. Create PyTorch Dataset
    print("\nPreparing dataset...")
    preprocessor = AudioPreprocessor(
        sample_rate=44100,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fixed_length=256
    )
    dataset = IntervalsDataset(
        data_dir=str(data_dir),
        transform=preprocessor.melspectrogram
    )
    
    # Split dataset
    torch_generator = torch.Generator().manual_seed(42)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch_generator
    )
    
    # Create data loaders
    num_workers = 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 3. Create and train model
    print("\nTraining model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = IntervalCNN(
        n_mels=128,
        n_intervals=len(interval_generator.intervals.INTERVALS)
    ).to(device)
    
    trainer = IntervalTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        checkpoint_dir=str(checkpoints_dir)
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        early_stopping_patience=10
    )
    
    # Save training history
    with open(stats_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # 4. Evaluate model
    print("\nEvaluating model...")
    evaluator = IntervalEvaluator(model, device=device)
    results = evaluator.evaluate(
        test_loader,
        class_names=list(interval_generator.intervals.INTERVALS.keys())
    )
    
    # Save evaluation results
    with open(stats_dir / 'evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }
        json.dump(results_serializable, f, indent=4)
    
    # Print and save evaluation metrics
    metrics_summary = ["Test Set Results:"]
    metrics_summary.append(f"Accuracy: {results['accuracy']:.2f}%")
    for interval, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            metrics_summary.extend([
                f"\n{interval}:",
                f"  Precision: {metrics['precision']:.2f}",
                f"  Recall: {metrics['recall']:.2f}",
                f"  F1-score: {metrics['f1-score']:.2f}"
            ])
    
    with open(stats_dir / 'metrics_summary.txt', 'w') as f:
        f.write('\n'.join(metrics_summary))
    
    # 5. Save visualizations
    print("\nSaving visualizations...")
    # Confusion matrix
    plt.figure(figsize=(12, 8))
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=list(interval_generator.intervals.INTERVALS.keys())
    )
    plt.savefig(plots_dir / 'confusion_matrix.png')
    plt.close()
    
    # Training history
    plt.figure(figsize=(15, 5))
    evaluator.plot_training_history(history)
    plt.savefig(plots_dir / 'training_history.png')
    plt.close()
    
    # 6. Save audio visualizations
    print("\nSaving audio visualizations...")
    audio_utils = AudioUtils()
    visualizer = AudioVisualizer()
    
    sample_files = list(data_dir.glob("*.wav"))
    if sample_files:
        visualized_intervals = set()
        for sample_file in sample_files:
            interval_name = sample_file.stem.split('_')[2]
            if interval_name not in visualized_intervals:
                audio_data, sr = audio_utils.load_audio(str(sample_file))
                plt.figure(figsize=(12, 12))
                visualizer.plot_multiple_features(audio_data)
                plt.savefig(plots_dir / f'audio_features_{interval_name}.png')
                plt.close()
                visualized_intervals.add(interval_name)
                if len(visualized_intervals) >= 3:  # Limit to 3 examples
                    break
    
    # Save training summary
    summary = [
        "\nTraining Summary:",
        f"Total samples: {len(dataset)}",
        f"Training samples: {len(train_dataset)}",
        f"Validation samples: {len(val_dataset)}",
        f"Test samples: {len(test_dataset)}",
        f"Final validation accuracy: {history['val_acc'][-1]:.2f}%",
        f"Best validation accuracy: {max(history['val_acc']):.2f}%",
        f"Model checkpoints saved in: {checkpoints_dir}",
        f"Plots saved in: {plots_dir}",
        f"Statistics saved in: {stats_dir}"
    ]
    
    with open(stats_dir / 'training_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))
    print(f"\nExperiment results saved in: {run_dir}")

if __name__ == "__main__":
    main()