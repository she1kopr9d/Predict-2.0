from data_fetcher import DataFetcher
from model import TimeSeriesPredictor
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configure GPU memory growth to prevent memory issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU")

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
    print("6. Predict next number")
    print("7. Analyze predictions with statistics")
    print("8. Analyze predictions with saving and graph")
    print("0. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (0-8): "))
            if 0 <= choice <= 8:
                return choice
            print("Please enter a number between 0 and 8")
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
    best_accuracy_above_2 = 0
    best_accuracy_above_1_5 = 0
    best_val_loss = float('inf')

    # Enable mixed precision for better GPU performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    while True:
        print(f"\nStarting training iteration {version//3 + 1}...")
        best_iteration_mae = float('inf')
        best_iteration_version = None
        best_points_above = float('inf')

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
                # Recompile the model with adjusted learning rate and mixed precision
                predictor.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
            else:
                predictor.create_model(input_shape=(X_train.shape[1], 1))
            
            # Train with early stopping to prevent overfitting
            history = predictor.train_model(X_train_combined, y_train_combined, X_test, y_test, epochs=30)
            
            # Evaluate
            stats = predictor.evaluate_model(X_test, y_test)
            current_mae = stats['mae']
            
            # Calculate points above line
            predictions = predictor.predict(X_test)
            points_above = np.sum(predictions.flatten() > y_test)
            
            # Calculate accuracy for new thresholds
            accuracy_above_2 = np.mean((predictions.flatten() > 2) == (y_test > 2))
            accuracy_above_1_5 = np.mean((predictions.flatten() > 1.5) == (y_test > 1.5))
            
            # Get validation loss from history
            val_loss = min(history.history['val_loss'])
            
            print(f"Version {current_version} - Test Loss: {stats['loss']:.4f}, MAE: {current_mae:.4f}")
            print(f"Points above line: {points_above}")
            print(f"Accuracy above 2: {accuracy_above_2:.4f}")
            print(f"Accuracy above 1.5: {accuracy_above_1_5:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save if better than current iteration best (consider multiple metrics)
            if (accuracy_above_2 > best_accuracy_above_2 or 
                (accuracy_above_2 == best_accuracy_above_2 and accuracy_above_1_5 > best_accuracy_above_1_5) or
                (accuracy_above_2 == best_accuracy_above_2 and accuracy_above_1_5 == best_accuracy_above_1_5 and val_loss < best_val_loss)):
                best_iteration_mae = current_mae
                best_points_above = points_above
                best_iteration_version = current_version
                best_accuracy_above_2 = accuracy_above_2
                best_accuracy_above_1_5 = accuracy_above_1_5
                best_val_loss = val_loss
                predictor.save_model(current_version)

        # Check if we improved over base model
        if (best_accuracy_above_2 > np.mean((predictor.predict(X_test).flatten() > 2) == (y_test > 2)) or
            (best_accuracy_above_2 == np.mean((predictor.predict(X_test).flatten() > 2) == (y_test > 2)) and
             best_accuracy_above_1_5 > np.mean((predictor.predict(X_test).flatten() > 1.5) == (y_test > 1.5)))):
            improvement_count += 1
            no_improvement_count = 0
            print(f"\nImprovement found!")
            print(f"Accuracy above 2: {best_accuracy_above_2:.4f}")
            print(f"Accuracy above 1.5: {best_accuracy_above_1_5:.4f}")
            print(f"Validation Loss: {best_val_loss:.4f}")
            
            # Save as new best model
            os.rename(f'model_v{best_iteration_version}.keras', 'best_model.keras')
            predictor.load_model('best_model.keras')
            
            # Show predictions with new best model
            predictions = predictor.predict(X_test)
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.axhline(y=2, color='r', linestyle='--', label='Threshold 2')
            plt.axhline(y=1.5, color='g', linestyle='--', label='Threshold 1.5')
            plt.title(f'Time Series Prediction (New Best Model)\nAccuracy above 2: {best_accuracy_above_2:.4f}, above 1.5: {best_accuracy_above_1_5:.4f}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        else:
            no_improvement_count += 1
            print(f"\nNo improvement in this iteration.")
            print(f"Best accuracy above 2: {best_accuracy_above_2:.4f}")
            print(f"Best accuracy above 1.5: {best_accuracy_above_1_5:.4f}")
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
    print(f"Final accuracy above 2: {best_accuracy_above_2:.4f}")
    print(f"Final accuracy above 1.5: {best_accuracy_above_1_5:.4f}")
    print(f"Final validation loss: {best_val_loss:.4f}")

def predict_next_number(predictor: TimeSeriesPredictor, data_fetcher: DataFetcher):
    """Fetch latest data and predict the next number."""
    if predictor.model is None:
        print("No model loaded. Please load or create a model first.")
        return

    # Ask user how many last numbers to use
    while True:
        try:
            sequence_length = int(input("\nHow many last numbers to use for prediction? (default is 30): ") or "30")
            if sequence_length > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    print("\nStarting automatic prediction updates...")
    print("Press Ctrl+C to stop")
    
    try:
        # Store initial data length
        initial_data = data_fetcher.fetch_data()
        predictor.last_prediction_data = initial_data.copy()
        
        while True:
            try:
                # Get latest data
                data = data_fetcher.fetch_data()
                
                # Check if we have new data
                if len(data) > len(predictor.last_prediction_data):
                    # New number has appeared - this was our previous prediction
                    new_number = data[-1]
                    print(f"\nNew number appeared: {new_number:.2f}")
                    print(f"This was our previous prediction: {predictor.last_prediction:.2f}")
                    print(f"Difference: {abs(new_number - predictor.last_prediction):.2f}")
                
                # Store current data and make new prediction
                predictor.last_prediction_data = data.copy()
                
                # Prepare the last sequence for prediction
                last_sequence = data[-sequence_length:]
                last_sequence = np.clip(last_sequence, 1, 10)  # Normalize to 1-10 range
                
                # Reshape data to match model's expected input shape
                X = last_sequence.reshape(1, sequence_length, 1)
                
                # Make prediction
                prediction = predictor.predict(X)[0][0]
                prediction = np.clip(prediction, 1, 10)  # Ensure prediction is in 1-10 range
                predictor.last_prediction = prediction  # Store prediction for next comparison
                
                print(f"\nLast {sequence_length} numbers: {last_sequence}")
                print(f"Predicted next number: {prediction:.2f}")
                
                # Wait for 5 seconds before next update
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\nStopping automatic updates...")
                break
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Data shape: {X.shape if 'X' in locals() else 'Not created'}")
        print(f"Last sequence shape: {last_sequence.shape if 'last_sequence' in locals() else 'Not created'}")

def analyze_predictions(predictor: TimeSeriesPredictor, data_fetcher: DataFetcher):
    """Analyze predictions with detailed statistics."""
    if predictor.model is None:
        print("No model loaded. Please load or create a model first.")
        return

    # Initialize statistics
    predictor.prediction_stats = {
        'total_predictions': 0,
        'predictions_above': 0,
        'predictions_below': 0,
        'exact_predictions': 0,
        'total_error': 0,
        'last_10_errors': []
    }

    # Ask user how many last numbers to use
    while True:
        try:
            sequence_length = int(input("\nHow many last numbers to use for prediction? (default is 30): ") or "30")
            if sequence_length > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    print("\nStarting prediction analysis...")
    print("Press Ctrl+C to stop")
    
    try:
        # Store initial data length
        initial_data = data_fetcher.fetch_data()
        predictor.last_prediction_data = initial_data.copy()
        
        while True:
            try:
                # Get latest data
                data = data_fetcher.fetch_data()
                
                # Check if we have new data
                if len(data) > len(predictor.last_prediction_data):
                    # New number has appeared - analyze previous prediction
                    new_number = data[-1]
                    error = abs(new_number - predictor.last_prediction)
                    
                    # Update statistics
                    predictor.prediction_stats['total_predictions'] += 1
                    predictor.prediction_stats['total_error'] += error
                    predictor.prediction_stats['last_10_errors'].append(error)
                    if len(predictor.prediction_stats['last_10_errors']) > 10:
                        predictor.prediction_stats['last_10_errors'].pop(0)
                    
                    if new_number > predictor.last_prediction:
                        predictor.prediction_stats['predictions_below'] += 1
                    elif new_number < predictor.last_prediction:
                        predictor.prediction_stats['predictions_above'] += 1
                    else:
                        predictor.prediction_stats['exact_predictions'] += 1
                    
                    # Print analysis
                    print("\n" + "="*50)
                    print(f"New number appeared: {new_number:.2f}")
                    print(f"Previous prediction: {predictor.last_prediction:.2f}")
                    print(f"Error: {error:.2f}")
                    print("\nStatistics:")
                    print(f"Total predictions: {predictor.prediction_stats['total_predictions']}")
                    print(f"Predictions above actual: {predictor.prediction_stats['predictions_above']}")
                    print(f"Predictions below actual: {predictor.prediction_stats['predictions_below']}")
                    print(f"Exact predictions: {predictor.prediction_stats['exact_predictions']}")
                    print(f"Average error: {predictor.prediction_stats['total_error']/predictor.prediction_stats['total_predictions']:.2f}")
                    print(f"Last 10 errors: {[f'{e:.2f}' for e in predictor.prediction_stats['last_10_errors']]}")
                    print("="*50)
                
                # Store current data and make new prediction
                predictor.last_prediction_data = data.copy()
                
                # Prepare the last sequence for prediction
                last_sequence = data[-sequence_length:]
                last_sequence = np.clip(last_sequence, 1, 10)  # Normalize to 1-10 range
                
                # Reshape data to match model's expected input shape
                X = last_sequence.reshape(1, sequence_length, 1)
                
                # Make prediction
                prediction = predictor.predict(X)[0][0]
                prediction = np.clip(prediction, 1, 10)  # Ensure prediction is in 1-10 range
                predictor.last_prediction = prediction  # Store prediction for next comparison
                
                print(f"\nLast {sequence_length} numbers: {last_sequence}")
                print(f"Predicted next number: {prediction:.2f}")
                
                # Wait for 5 seconds before next update
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\nStopping prediction analysis...")
                break
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Data shape: {X.shape if 'X' in locals() else 'Not created'}")
        print(f"Last sequence shape: {last_sequence.shape if 'last_sequence' in locals() else 'Not created'}")

def analyze_and_save_predictions(predictor: TimeSeriesPredictor, data_fetcher: DataFetcher):
    """Analyze predictions with detailed statistics, save to file and show live graph."""
    if predictor.model is None:
        print("No model loaded. Please load or create a model first.")
        return

    # Initialize statistics
    predictor.prediction_stats = {
        'total_predictions': 0,
        'predictions_above': 0,
        'predictions_below': 0,
        'exact_predictions': 0,
        'total_error': 0,
        'last_10_errors': [],
        'all_predictions': [],
        'all_actuals': [],
        'all_errors': [],
        'points_above_line': 0,
        'points_below_line': 0,
        'points_on_line': 0
    }

    # Create or clear the log file
    with open('prediction_log.txt', 'w') as f:
        f.write("Time,Actual,Prediction,Error,Position\n")

    # Ask user how many last numbers to use
    while True:
        try:
            sequence_length = int(input("\nHow many last numbers to use for prediction? (default is 30): ") or "30")
            if sequence_length > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    print("\nStarting prediction analysis with saving...")
    print("Press Ctrl+C to stop")
    
    # Initialize the plot with four subplots
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(7.5, 6))  # Half the original size
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Add figure size adjustment controls
    fig.subplots_adjust(bottom=0.2)
    ax_width = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_height = plt.axes([0.2, 0.05, 0.65, 0.03])
    
    width_slider = plt.Slider(ax_width, 'Width', 5, 10, valinit=7.5)  # Half the original range
    height_slider = plt.Slider(ax_height, 'Height', 4, 8, valinit=6)  # Half the original range
    
    def update_figure_size(val):
        fig.set_size_inches(width_slider.val, height_slider.val)
        fig.canvas.draw_idle()
    
    width_slider.on_changed(update_figure_size)
    height_slider.on_changed(update_figure_size)
    
    # Initialize plot lines and scatter
    line1, = ax1.plot([], [], 'b-', label='Actual')
    line2, = ax1.plot([], [], 'r--', label='Predicted')
    line3, = ax2.plot([], [], 'g-')
    scatter_above, = ax3.plot([], [], 'ro', alpha=0.5, label='Above Line')
    scatter_below, = ax3.plot([], [], 'go', alpha=0.5, label='Below Line')
    scatter_on, = ax3.plot([], [], 'bo', alpha=0.5, label='On Line')
    line4, = ax3.plot([], [], 'k-', label='Perfect Prediction')
    line5, = ax4.plot([], [], 'b-', label='Actual')
    line6, = ax4.plot([], [], 'r--', label='Predicted')
    scatter2, = ax4.plot([], [], 'go', alpha=0.5, label='Deviation Points')
    
    # Set titles and labels
    ax1.set_title('Actual vs Predicted Values')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Prediction Error Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error')
    ax2.grid(True)
    
    ax3.set_title('Actual vs Predicted Scatter Plot')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('Deviation Analysis')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    # Set perfect prediction line
    ax3.plot([1, 10], [1, 10], 'k-', label='Perfect Prediction')
    
    plt.tight_layout()
    
    try:
        # Store initial data length
        initial_data = data_fetcher.fetch_data()
        predictor.last_prediction_data = initial_data.copy()
        
        while True:
            try:
                # Get latest data
                data = data_fetcher.fetch_data()
                
                # Check if we have new data
                if len(data) > len(predictor.last_prediction_data):
                    # New number has appeared - analyze previous prediction
                    new_number = data[-1]
                    # Normalize actual value to 1-10 range
                    normalized_actual = np.clip(new_number, 1, 10)
                    error = abs(normalized_actual - predictor.last_prediction)
                    
                    # Determine position relative to perfect prediction line
                    position = "above" if predictor.last_prediction > normalized_actual else "below" if predictor.last_prediction < normalized_actual else "on"
                    
                    # Update position statistics
                    if position == "above":
                        predictor.prediction_stats['points_above_line'] += 1
                    elif position == "below":
                        predictor.prediction_stats['points_below_line'] += 1
                    else:
                        predictor.prediction_stats['points_on_line'] += 1
                    
                    # Update statistics
                    predictor.prediction_stats['total_predictions'] += 1
                    predictor.prediction_stats['total_error'] += error
                    predictor.prediction_stats['last_10_errors'].append(error)
                    if len(predictor.prediction_stats['last_10_errors']) > 10:
                        predictor.prediction_stats['last_10_errors'].pop(0)
                    
                    predictor.prediction_stats['all_predictions'].append(predictor.last_prediction)
                    predictor.prediction_stats['all_actuals'].append(normalized_actual)
                    predictor.prediction_stats['all_errors'].append(error)
                    
                    if normalized_actual > predictor.last_prediction:
                        predictor.prediction_stats['predictions_below'] += 1
                    elif normalized_actual < predictor.last_prediction:
                        predictor.prediction_stats['predictions_above'] += 1
                    else:
                        predictor.prediction_stats['exact_predictions'] += 1
                    
                    # Save to log file
                    with open('prediction_log.txt', 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{normalized_actual:.2f},{predictor.last_prediction:.2f},{error:.2f},{position}\n")
                    
                    # Print analysis
                    print("\n" + "="*50)
                    print(f"New number appeared: {new_number:.2f} (normalized to {normalized_actual:.2f})")
                    print(f"Previous prediction: {predictor.last_prediction:.2f}")
                    print(f"Error: {error:.2f}")
                    print(f"Position relative to line: {position}")
                    print("\nStatistics:")
                    print(f"Total predictions: {predictor.prediction_stats['total_predictions']}")
                    print(f"Points above line: {predictor.prediction_stats['points_above_line']}")
                    print(f"Points below line: {predictor.prediction_stats['points_below_line']}")
                    print(f"Points on line: {predictor.prediction_stats['points_on_line']}")
                    print(f"Predictions above actual: {predictor.prediction_stats['predictions_above']}")
                    print(f"Predictions below actual: {predictor.prediction_stats['predictions_below']}")
                    print(f"Exact predictions: {predictor.prediction_stats['exact_predictions']}")
                    print(f"Average error: {predictor.prediction_stats['total_error']/predictor.prediction_stats['total_predictions']:.2f}")
                    print(f"Last 10 errors: {[f'{e:.2f}' for e in predictor.prediction_stats['last_10_errors']]}")
                    print("="*50)
                    
                    # Update plot data
                    x = range(len(predictor.prediction_stats['all_actuals']))
                    
                    # Update lines
                    line1.set_data(x, predictor.prediction_stats['all_actuals'])
                    line2.set_data(x, predictor.prediction_stats['all_predictions'])
                    line3.set_data(x, predictor.prediction_stats['all_errors'])
                    
                    # Update scatter plot with different colors for points above/below/on line
                    above_mask = np.array(predictor.prediction_stats['all_predictions']) > np.array(predictor.prediction_stats['all_actuals'])
                    below_mask = np.array(predictor.prediction_stats['all_predictions']) < np.array(predictor.prediction_stats['all_actuals'])
                    on_mask = np.array(predictor.prediction_stats['all_predictions']) == np.array(predictor.prediction_stats['all_actuals'])
                    
                    scatter_above.set_data(
                        np.array(predictor.prediction_stats['all_actuals'])[above_mask],
                        np.array(predictor.prediction_stats['all_predictions'])[above_mask]
                    )
                    scatter_below.set_data(
                        np.array(predictor.prediction_stats['all_actuals'])[below_mask],
                        np.array(predictor.prediction_stats['all_predictions'])[below_mask]
                    )
                    scatter_on.set_data(
                        np.array(predictor.prediction_stats['all_actuals'])[on_mask],
                        np.array(predictor.prediction_stats['all_predictions'])[on_mask]
                    )
                    
                    # Update deviation analysis plot
                    line5.set_data(x, predictor.prediction_stats['all_actuals'])
                    line6.set_data(x, predictor.prediction_stats['all_predictions'])
                    
                    # Find points with significant deviation
                    significant_deviation = np.where(np.array(predictor.prediction_stats['all_errors']) > 1.0)[0]
                    if len(significant_deviation) > 0:
                        scatter2.set_data(significant_deviation, 
                                        [predictor.prediction_stats['all_actuals'][i] for i in significant_deviation])
                    
                    # Update axes limits
                    ax1.set_xlim(0, len(x))
                    ax1.set_ylim(0, 11)
                    ax2.set_xlim(0, len(x))
                    ax2.set_ylim(0, max(predictor.prediction_stats['all_errors']) * 1.1 if predictor.prediction_stats['all_errors'] else 1)
                    ax3.set_xlim(0, 11)
                    ax3.set_ylim(0, 11)
                    ax4.set_xlim(0, len(x))
                    ax4.set_ylim(0, 11)
                    
                    # Draw the plot
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                # Store current data and make new prediction
                predictor.last_prediction_data = data.copy()
                
                # Prepare the last sequence for prediction
                last_sequence = data[-sequence_length:]
                last_sequence = np.clip(last_sequence, 1, 10)  # Normalize to 1-10 range
                
                # Reshape data to match model's expected input shape
                X = last_sequence.reshape(1, sequence_length, 1)
                
                # Make prediction
                prediction = predictor.predict(X)[0][0]
                prediction = np.clip(prediction, 1, 10)  # Ensure prediction is in 1-10 range
                predictor.last_prediction = prediction  # Store prediction for next comparison
                
                print(f"\nLast {sequence_length} numbers: {last_sequence}")
                print(f"Predicted next number: {prediction:.2f}")
                
                # Wait for 5 seconds before next update
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\nStopping prediction analysis...")
                break
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Data shape: {X.shape if 'X' in locals() else 'Not created'}")
        print(f"Last sequence shape: {last_sequence.shape if 'last_sequence' in locals() else 'Not created'}")
    finally:
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close the plot

def main():
    # Initialize components
    data_fetcher = DataFetcher(sequence_length=30)
    predictor = TimeSeriesPredictor(sequence_length=30)

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
            
        elif choice == 6:
            predict_next_number(predictor, data_fetcher)
            
        elif choice == 7:
            analyze_predictions(predictor, data_fetcher)
            
        elif choice == 8:
            analyze_and_save_predictions(predictor, data_fetcher)

if __name__ == "__main__":
    main() 