from data_fetcher import DataFetcher
from model import TimeSeriesPredictor
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
from tensorflow.keras.models import Sequential

def train_iteration(predictor: TimeSeriesPredictor, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, version: int) -> dict:
    """Train one iteration of the model and return its statistics."""
    print(f"\nTraining model version {version}...")
    predictor.create_model(input_shape=(X_train.shape[1], 1))
    predictor.train_model(X_train, y_train, X_test, y_test, epochs=50)
    
    # Evaluate model
    stats = predictor.evaluate_model(X_test, y_test)
    print(f"Version {version} - Test Loss: {stats['loss']:.4f}, MAE: {stats['mae']:.4f}")
    
    # Save model and compare with best
    predictor.save_model(version)
    predictor.compare_and_save_best(version, stats)
    
    return stats

def show_menu():
    """Display the main menu and get user choice."""
    print("\n=== Time Series Prediction System ===")
    print("1. Create new model")
    print("2. Load specific model version")
    print("3. Start training with loaded model")
    print("4. Test loaded model")
    print("5. Test and compare all models")
    print("0. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (0-5): "))
            if 0 <= choice <= 5:
                return choice
            print("Please enter a number between 0 and 5")
        except ValueError:
            print("Please enter a valid number")

def get_available_versions():
    """Get list of available model versions."""
    model_files = glob.glob('model_v*.keras')
    versions = [int(f.split('_v')[1].split('.')[0]) for f in model_files]
    return sorted(versions)

def select_model_version():
    """Let user select a model version to load."""
    versions = get_available_versions()
    if not versions:
        print("No saved models found. Please create a new model first.")
        return None
    
    print("\nAvailable model versions:")
    for v in versions:
        print(f"Version {v}")
    
    while True:
        try:
            version = int(input("\nEnter version number to load: "))
            if version in versions:
                return version
            print(f"Please enter a valid version number from the list: {versions}")
        except ValueError:
            print("Please enter a valid number")

def test_all_models(predictor: TimeSeriesPredictor, X_test: np.ndarray, y_test: np.ndarray):
    """Test all available models and keep only the best one."""
    versions = get_available_versions()
    if not versions:
        print("No models found to test.")
        return

    print("\nTesting all available models...")
    results = []
    
    # Test each model
    for version in versions:
        model_path = f'model_v{version}.keras'
        predictor.load_model(model_path)
        predictions = predictor.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        rmse = np.sqrt(mse)
        
        results.append({
            'version': version,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'path': model_path
        })
        
        print(f"\nModel Version {version} Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    # Find best model based on MAE
    best_model = min(results, key=lambda x: x['mae'])
    print(f"\nBest model is Version {best_model['version']} with MAE: {best_model['mae']:.4f}")
    
    # Delete all models except the best one
    for result in results:
        if result['version'] != best_model['version']:
            try:
                os.remove(result['path'])
                print(f"Deleted model version {result['version']}")
            except OSError as e:
                print(f"Error deleting model version {result['version']}: {e}")
    
    # Save best model as the main model
    os.rename(best_model['path'], 'best_model.keras')
    print("\nBest model saved as 'best_model.keras'")
    
    # Load and show best model results
    predictor.load_model('best_model.keras')
    test_model(predictor, X_test, y_test)

def test_model(predictor: TimeSeriesPredictor, X_test: np.ndarray, y_test: np.ndarray):
    """Test the loaded model with different metrics and visualizations."""
    if predictor.model is None:
        print("No model loaded. Please load a model first.")
        return

    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions.flatten()) ** 2)
    mae = np.mean(np.abs(y_test - predictions.flatten()))
    rmse = np.sqrt(mse)
    
    print("\nModel Test Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot 2: Error Distribution
    plt.subplot(2, 2, 2)
    errors = y_test - predictions.flatten()
    plt.hist(errors, bins=50)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    # Plot 3: Scatter Plot
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot 4: Cumulative Error
    plt.subplot(2, 2, 4)
    cumulative_error = np.cumsum(np.abs(errors))
    plt.plot(cumulative_error)
    plt.title('Cumulative Absolute Error')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Error')
    
    plt.tight_layout()
    plt.show()

def create_new_model(predictor: TimeSeriesPredictor, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray):
    """Create and train initial model."""
    print("\nCreating new model...")
    predictor.create_model(input_shape=(X_train.shape[1], 1))
    predictor.train_model(X_train, y_train, X_test, y_test, epochs=50)
    stats = predictor.evaluate_model(X_test, y_test)
    print(f"Initial model - Test Loss: {stats['loss']:.4f}, MAE: {stats['mae']:.4f}")
    predictor.save_model(1)
    predictor.compare_and_save_best(1, stats)

def load_model(predictor: TimeSeriesPredictor):
    """Load a specific model version."""
    version = select_model_version()
    if version is None:
        return False
    
    model_path = f'model_v{version}.keras'
    print(f"\nLoading model version {version}...")
    predictor.load_model(model_path)
    return True

def start_training(predictor: TimeSeriesPredictor, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, data_fetcher: DataFetcher):
    """Start iterative training process using the best model as base."""
    if predictor.model is None:
        print("No model loaded. Please load or create a model first.")
        return

    # Load the best model if it exists
    if os.path.exists('best_model.keras'):
        print("\nLoading best model as base for training...")
        predictor.load_model('best_model.keras')
        base_mae = predictor.evaluate_model(X_test, y_test)['mae']
        print(f"Base model MAE: {base_mae:.4f}")
    else:
        print("\nNo best model found. Starting from scratch...")
        base_mae = float('inf')

    version = len(get_available_versions()) + 1
    improvement_count = 0
    no_improvement_count = 0
    max_no_improvement = 3  # Stop if no improvement after 3 iterations

    while True:
        print(f"\nStarting training iteration {version//3 + 1}...")
        best_iteration_mae = float('inf')
        best_iteration_version = None

        # Get current predictions and analyze errors
        current_predictions = predictor.predict(X_test)
        problematic_X, problematic_y = data_fetcher.analyze_errors(current_predictions, y_test)
        
        if len(problematic_X) > 0:
            print(f"\nFound {len(problematic_X)} problematic sequences for targeted training")
            # Combine problematic sequences with regular training data
            X_train_combined = np.vstack([X_train, problematic_X])
            y_train_combined = np.concatenate([y_train, problematic_y])
        else:
            X_train_combined = X_train
            y_train_combined = y_train

        # Train three versions
        for i in range(3):
            current_version = version + i
            print(f"\nTraining model version {current_version}...")
            
            # Create new model as a copy of the best model
            if os.path.exists('best_model.keras'):
                predictor.load_model('best_model.keras')
                # Clone the model to get a fresh copy
                model_config = predictor.model.get_config()
                weights = predictor.model.get_weights()
                predictor.model = Sequential.from_config(model_config)
                predictor.model.set_weights(weights)
                # Recompile the model
                predictor.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
            else:
                predictor.create_model(input_shape=(X_train.shape[1], 1))
            
            # Train with more epochs for fine-tuning
            predictor.train_model(X_train_combined, y_train_combined, X_test, y_test, epochs=30)
            
            # Evaluate
            stats = predictor.evaluate_model(X_test, y_test)
            current_mae = stats['mae']
            print(f"Version {current_version} - Test Loss: {stats['loss']:.4f}, MAE: {current_mae:.4f}")
            
            # Save if better than current iteration best
            if current_mae < best_iteration_mae:
                best_iteration_mae = current_mae
                best_iteration_version = current_version
                predictor.save_model(current_version)

        # Check if we improved over base model
        if best_iteration_mae < base_mae:
            improvement_count += 1
            no_improvement_count = 0
            print(f"\nImprovement found! New best MAE: {best_iteration_mae:.4f} (was {base_mae:.4f})")
            base_mae = best_iteration_mae
            
            # Save as new best model
            os.rename(f'model_v{best_iteration_version}.keras', 'best_model.keras')
            predictor.load_model('best_model.keras')
            
            # Show predictions with new best model
            predictions = predictor.predict(X_test)
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title(f'Time Series Prediction (New Best Model - MAE: {base_mae:.4f})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        else:
            no_improvement_count += 1
            print(f"\nNo improvement in this iteration. Best MAE remains: {base_mae:.4f}")
            if no_improvement_count >= max_no_improvement:
                print(f"\nNo improvement after {max_no_improvement} iterations. Stopping training.")
                break

        version += 3

        # Ask user if they want to continue
        response = input("\nDo you want to continue training? (y/n): ")
        if response.lower() != 'y':
            break

    print("\nTraining complete!")
    print(f"Total improvements: {improvement_count}")
    print(f"Final best MAE: {base_mae:.4f}")

def main():
    # Initialize components
    data_fetcher = DataFetcher()
    predictor = TimeSeriesPredictor(sequence_length=10)

    # Fetch and prepare data
    print("Fetching data...")
    data = data_fetcher.fetch_data()
    X, y = data_fetcher.prepare_sequences(data)
    X_train, X_test, y_train, y_test = data_fetcher.split_data(X, y)

    # Try to load the best model at startup
    if os.path.exists('best_model.keras'):
        print("\nLoading best model...")
        predictor.load_model('best_model.keras')
        stats = predictor.evaluate_model(X_test, y_test)
        print(f"Loaded model MAE: {stats['mae']:.4f}")
    else:
        print("\nNo best model found. You can create a new model using option 1.")

    while True:
        choice = show_menu()
        
        if choice == 0:
            print("Exiting program...")
            break
            
        elif choice == 1:
            create_new_model(predictor, X_train, y_train, X_test, y_test)
            
        elif choice == 2:
            if load_model(predictor):
                print("Model loaded successfully!")
                
        elif choice == 3:
            start_training(predictor, X_train, y_train, X_test, y_test, data_fetcher)
            
        elif choice == 4:
            test_model(predictor, X_test, y_test)
            
        elif choice == 5:
            test_all_models(predictor, X_test, y_test)

if __name__ == "__main__":
    main() 