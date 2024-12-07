from dataclasses import dataclass
from typing import Dict

@dataclass
class GenerationConfig:
    sample_rate: int = 44100
    duration: float = 1.0
    amplitude: float = 0.5
    note_gap: float = 0.1  # Gap between notes in seconds
    
    # Frequency ranges
    min_freq: float = 220.0  # A3
    max_freq: float = 440.0  # A4

@dataclass
class IntervalDefinitions:
    INTERVALS: Dict[str, int] = None
    
    def __post_init__(self):
        self.INTERVALS = {
            'minor_second': 1,
            'major_second': 2,
            'minor_third': 3,
            'major_third': 4,
            'perfect_fourth': 5,
            'tritone': 6,
            'perfect_fifth': 7,
            'minor_sixth': 8,
            'major_sixth': 9,
            'minor_seventh': 10,
            'major_seventh': 11,
            'octave': 12
        }