import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os
from typing import Tuple, Optional, Dict
import json

class TimeSeriesPredictor:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model: Optional[Sequential] = None
        self.best_model_path = 'best_model.keras'
        self.model_stats_path = 'model_stats.json'
        self.model_stats: Dict = self._load_stats()

    def _load_stats(self) -> Dict:
        """Load model statistics from file."""
        if os.path.exists(self.model_stats_path):
            with open(self.model_stats_path, 'r') as f:
                return json.load(f)
        return {'best_mae': float('inf'), 'versions': []}

    def _save_stats(self):
        """Save model statistics to file."""
        with open(self.model_stats_path, 'w') as f:
            json.dump(self.model_stats, f, indent=4)

    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create a new LSTM model for time series prediction."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        self.model = model
        return model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model with GPU support."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
        
        # Convert data to float32 for GPU compatibility
        X_train = X_train.astype('float32')
        y_train = y_train.astype('float32')
        X_val = X_val.astype('float32')
        y_val = y_val.astype('float32')
        
        # Force training on GPU
        with tf.device('/device:GPU:0'):
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
        
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not created or loaded.")
        return self.model.predict(X)

    def save_model(self, version: int):
        """Save the model to a file with version number."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = f'model_v{version}.keras'
        self.model.save(model_path, save_format='keras')
        return model_path

    def load_model(self, filepath: str):
        """Load a model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = load_model(filepath)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model's performance on test data."""
        if self.model is None:
            raise ValueError("Model not created or loaded.")
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {'loss': float(loss), 'mae': float(mae)}

    def compare_and_save_best(self, version: int, stats: dict):
        """Compare current model with best model and update if better."""
        current_mae = stats['mae']
        
        # Save version stats
        self.model_stats['versions'].append({
            'version': version,
            'mae': current_mae,
            'loss': stats['loss']
        })
        
        # Update best model if current is better
        if current_mae < self.model_stats['best_mae']:
            self.model_stats['best_mae'] = current_mae
            self.model.save(self.best_model_path, save_format='keras')
            print(f"New best model saved! MAE: {current_mae:.4f}")
        
        self._save_stats() 