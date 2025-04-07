import requests
import numpy as np
from typing import Tuple, List

class DataFetcher:
    def __init__(self, api_url: str = "http://92.255.175.128:8001/get_x/", sequence_length: int = 10):
        self.api_url = api_url
        self.sequence_length = sequence_length

    def fetch_data(self) -> np.ndarray:
        """Fetch data from the API and return as numpy array."""
        response = requests.get(self.api_url)
        if response.status_code == 200:
            data = response.json()['data']
            return np.array(data)
        else:
            raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    def analyze_errors(self, predictions: np.ndarray, actual: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze prediction errors and identify problematic sequences.
        
        Args:
            predictions: Model predictions
            actual: Actual values
            threshold: Error threshold to consider a prediction problematic
            
        Returns:
            Tuple of (problematic_sequences, problematic_targets) containing sequences
            where the model made significant errors
        """
        # Calculate absolute errors
        errors = np.abs(predictions - actual)
        
        # Find indices where error is above threshold
        problematic_indices = np.where(errors > threshold)[0]
        
        # Get sequences and targets for problematic predictions
        problematic_sequences = []
        problematic_targets = []
        
        for idx in problematic_indices:
            if idx >= self.sequence_length:
                seq = actual[idx - self.sequence_length:idx]
                target = actual[idx]
                problematic_sequences.append(seq)
                problematic_targets.append(target)
        
        return np.array(problematic_sequences), np.array(problematic_targets)

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training with values normalized to 1-10 range."""
        # Normalize all values to be between 1 and 10
        data = np.clip(data, 1, 10)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test 