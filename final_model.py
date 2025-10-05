"""
Final Exoplanet Classification Model
Optimized Gradient Boosting model for UI integration with prediction and retraining capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import logging
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

from data_processing import process_data

class FinalExoplanetModel:
    """
    Final Gradient Boosting model for exoplanet classification with UI integration capabilities.
    """
    
    def __init__(self, hyperparameters: Optional[Dict] = None, output_dir: str = "model_outputs"):
        """
        Initialize the model with Gradient Boosting and optional hyperparameters.
        
        Args:
            hyperparameters: Dictionary of model hyperparameters
            output_dir: Directory to save all outputs and logs
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.exoplanet_names = None  # Store exoplanet names for output
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # Optimized hyperparameters for Gradient Boosting
        self.hyperparameters = hyperparameters or {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Setup comprehensive logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging to both file and console."""
        # Create timestamp for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, "logs", f"exoplanet_model_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Log session start
        logging.info("=" * 60)
        logging.info(f"EXOPLANET MODEL SESSION STARTED - {datetime.now()}")
        logging.info("=" * 60)
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Log file: {log_file}")
    
    def save_predictions_to_file(self, predictions: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save predictions to JSON and CSV files.
        
        Args:
            predictions: List of prediction dictionaries
            filename: Optional custom filename
            
        Returns:
            Path to saved files
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}"
        
        # Save as JSON (for UI consumption)
        json_path = os.path.join(self.output_dir, "predictions", f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        # Save as CSV (for analysis)
        csv_data = []
        for pred in predictions:
            row = {
                'exoplanet_name': pred['exoplanet_name'],
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence']
            }
            # Add probability columns
            for class_name, prob in pred['class_probabilities'].items():
                row[f'prob_{class_name}'] = prob
            csv_data.append(row)
        
        csv_path = os.path.join(self.output_dir, "predictions", f"{filename}.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        logging.info(f"Predictions saved to:")
        logging.info(f"  JSON: {json_path}")
        logging.info(f"  CSV: {csv_path}")
        
        return json_path
    
    def save_model_summary(self, training_results: Dict[str, Any]) -> str:
        """
        Save comprehensive model summary and results.
        
        Args:
            training_results: Results from model training
            
        Returns:
            Path to summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, "logs", f"model_summary_{timestamp}.json")
        
        summary = {
            "timestamp": timestamp,
            "model_type": "Gradient Boosting Classifier",
            "hyperparameters": self.hyperparameters,
            "training_results": training_results,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "classes": list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else [],
            "status": "trained" if self.is_trained else "not_trained"
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"Model summary saved to: {summary_path}")
        return summary_path
    
    def load_and_prepare_data(self, input_file: str = 'cumulative_2025.10.04_07.24.46.csv') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data using the data processing function.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Tuple of (features DataFrame, target Series, exoplanet names Series)
        """
        logging.info("Loading and preparing data with data processing pipeline...")
        
        # Load raw data first to get exoplanet names before processing
        raw_df = pd.read_csv(input_file, comment='#')
        exoplanet_names = raw_df.get('kepoi_name', pd.Series(range(len(raw_df))))
        
        # Get processed data
        df = process_data(input_file=input_file, save_csv=False, setup_logging=False)
        
        # Check if target column exists
        if 'koi_disposition' not in df.columns:
            raise ValueError("Target column 'koi_disposition' not found in processed data")
        
        # Separate features and target
        X = df.drop('koi_disposition', axis=1)
        y = df['koi_disposition']
        
        # Store feature names and exoplanet names
        self.feature_names = X.columns.tolist()
        self.exoplanet_names = exoplanet_names
        
        # Handle any remaining categorical variables in features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        logging.info(f"Data loaded successfully!")
        logging.info(f"Shape: {X.shape}")
        logging.info(f"Classes: {self.label_encoder.classes_}")
        logging.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, pd.Series(y_encoded), exoplanet_names
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the Gradient Boosting model.
        
        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        logging.info("Training Gradient Boosting model...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if tune_hyperparameters:
            logging.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'min_samples_split': [5, 10, 15],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.hyperparameters = grid_search.best_params_
            
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use provided hyperparameters
            self.model = GradientBoostingClassifier(**self.hyperparameters)
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val_scaled)
        val_probabilities = self.model.predict_proba(X_val_scaled)
        
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average='weighted')
        
        self.is_trained = True
        
        results = {
            'validation_accuracy': val_accuracy,
            'validation_f1': val_f1,
            'hyperparameters': self.hyperparameters,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        logging.info(f"Model trained successfully!")
        logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logging.info(f"Validation F1 Score: {val_f1:.4f}")
        
        return results
    
    def predict(self, data_input, input_type: str = 'file', save_outputs: bool = True) -> List[Dict[str, Any]]:
        """
        Make predictions on new data with exoplanet names, classes, and probabilities.
        
        Args:
            data_input: Either file path (str) or DataFrame
            input_type: 'file' or 'dataframe'
            save_outputs: Whether to save outputs to files (default: True)
            
        Returns:
            List of dictionaries with exoplanet_name, predicted_class, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if input_type == 'file':
            # Load and process new data
            X, _, exoplanet_names = self.load_and_prepare_data(data_input)
        else:
            # Process DataFrame input
            df_processed = process_data(save_csv=False, setup_logging=False)  # This needs the DataFrame input
            X = df_processed.drop('koi_disposition', axis=1, errors='ignore')
            exoplanet_names = data_input.get('kepoi_name', pd.Series(range(len(X))))
            
            # Handle categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Ensure features match training data
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions_encoded = self.model.predict(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Format results for UI
        results = []
        for i in range(len(predictions)):
            exoplanet_result = {
                'exoplanet_name': str(exoplanet_names.iloc[i]) if hasattr(exoplanet_names, 'iloc') else f"Object_{i}",
                'predicted_class': predictions[i],
                'class_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.label_encoder.classes_, probabilities[i])
                },
                'confidence': float(np.max(probabilities[i]))
            }
            results.append(exoplanet_result)
        
        logging.info(f"Predictions completed for {len(results)} objects")
        
        # Conditionally save predictions to files
        if save_outputs:
            self.save_predictions_to_file(results)
        
        # Log prediction summary
        class_counts = {}
        for result in results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if save_outputs:
            logging.info("Prediction Summary:")
            for class_name, count in class_counts.items():
                percentage = (count / len(results)) * 100
                logging.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return results
    
    def predict_simple(self, data_input, input_type: str = 'file') -> List[Dict[str, Any]]:
        """
        Simple prediction method for UI - returns results directly in memory without saving.
        
        Args:
            data_input: Either file path (str) or DataFrame
            input_type: 'file' or 'dataframe'
            
        Returns:
            List of dictionaries with predictions ready for UI consumption
        """
        return self.predict(data_input, input_type, save_outputs=False)
    
    def predict_with_simple_format(self, data_input, input_type: str = 'file') -> Dict[str, Any]:
        """
        Make predictions and return in a simple format perfect for UI display.
        
        Args:
            data_input: Either file path (str) or DataFrame
            input_type: 'file' or 'dataframe'
            
        Returns:
            Dictionary with organized prediction data for UI
        """
        # Get predictions without saving
        predictions = self.predict(data_input, input_type, save_outputs=False)
        
        # Organize data for UI
        ui_data = {
            'total_predictions': len(predictions),
            'predictions': predictions,
            'class_summary': {},
            'high_confidence_count': 0,
            'exoplanet_names': [],
            'predicted_classes': [],
            'confidences': []
        }
        
        # Calculate summary statistics
        class_counts = {}
        high_confidence_threshold = 0.8
        
        for pred in predictions:
            class_name = pred['predicted_class']
            confidence = pred['confidence']
            
            # Count classes
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Count high confidence predictions
            if confidence > high_confidence_threshold:
                ui_data['high_confidence_count'] += 1
            
            # Extract simple lists for easy UI access
            ui_data['exoplanet_names'].append(pred['exoplanet_name'])
            ui_data['predicted_classes'].append(pred['predicted_class'])
            ui_data['confidences'].append(pred['confidence'])
        
        # Add class summary with percentages
        for class_name, count in class_counts.items():
            percentage = (count / len(predictions)) * 100
            ui_data['class_summary'][class_name] = {
                'count': count,
                'percentage': round(percentage, 1)
            }
        
        return ui_data
    
    def get_predictions_as_lists(self, data_input, input_type: str = 'file') -> Tuple[List[str], List[str], List[float], List[Dict[str, float]]]:
        """
        Get predictions as simple lists - perfect for direct UI integration.
        
        Args:
            data_input: Either file path (str) or DataFrame
            input_type: 'file' or 'dataframe'
            
        Returns:
            Tuple of (exoplanet_names, predicted_classes, confidences, all_probabilities)
        """
        predictions = self.predict(data_input, input_type, save_outputs=False)
        
        exoplanet_names = [pred['exoplanet_name'] for pred in predictions]
        predicted_classes = [pred['predicted_class'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        all_probabilities = [pred['class_probabilities'] for pred in predictions]
        
        return exoplanet_names, predicted_classes, confidences, all_probabilities
    
    def retrain_model(self, new_data_file: str, hyperparameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retrain the model on new data.
        
        Args:
            new_data_file: Path to new training data
            hyperparameters: Optional new hyperparameters
            
        Returns:
            Training results dictionary
        """
        logging.info("Retraining model on new data...")
        
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        # Load new data
        X, y, _ = self.load_and_prepare_data(new_data_file)
        
        # Retrain model
        results = self.train_model(X, y, tune_hyperparameters=False)
        
        logging.info("Model retrained successfully!")
        return results
    
    def plot_prediction_frequency(self, predictions: List[Dict[str, Any]], save_path: str = None) -> plt.Figure:
        """
        Create a frequency plot of prediction classes.
        
        Args:
            predictions: List of prediction dictionaries
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract predicted classes
        predicted_classes = [pred['predicted_class'] for pred in predictions]
        class_counts = pd.Series(predicted_classes).value_counts()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        bars = ax.bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)])
        ax.set_title('Frequency of Predicted Exoplanet Classes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Add percentage labels
        total = sum(class_counts.values)
        for i, (class_name, count) in enumerate(class_counts.items()):
            percentage = (count / total) * 100
            ax.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Use output directory if save_path not specified
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, "plots", f"prediction_frequency_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logging.info(f"Prediction frequency plot saved to {save_path}")
        return fig
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None) -> plt.Figure:
        """
        Create a feature importance plot.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot feature importance")
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], color='skyblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features for Exoplanet Classification', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        # Use output directory if save_path not specified
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, "plots", f"feature_importance_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logging.info(f"Feature importance plot saved to {save_path}")
        return fig
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath: str = 'final_exoplanet_model.joblib') -> None:
        """Save the trained model and all preprocessing objects."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'exoplanet_names': self.exoplanet_names
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.hyperparameters = model_data.get('hyperparameters', {})
        self.exoplanet_names = model_data.get('exoplanet_names', None)
        self.is_trained = True
        
        logging.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        return {
            "status": "Model trained",
            "model_type": "Gradient Boosting Classifier",
            "hyperparameters": self.hyperparameters,
            "feature_count": len(self.feature_names),
            "classes": list(self.label_encoder.classes_),
            "feature_names": self.feature_names[:10]  # First 10 features
        }

def main():
    """Example usage of the FinalExoplanetModel."""
    
    # Custom hyperparameters for demonstration
    custom_hyperparameters = {
        'n_estimators': 250,
        'learning_rate': 0.12,
        'max_depth': 7,
        'min_samples_split': 8,
        'min_samples_leaf': 3,
        'subsample': 0.85,
        'random_state': 42
    }
    
    # Initialize model with output directory
    model = FinalExoplanetModel(
        hyperparameters=custom_hyperparameters,
        output_dir="exoplanet_model_outputs"
    )
    
    try:
        # Load and train
        print("=== TRAINING MODEL ===")
        X, y, exoplanet_names = model.load_and_prepare_data()
        training_results = model.train_model(X, y)
        
        # Save model summary
        model.save_model_summary(training_results)
        
        print(f"Training completed!")
        print(f"Validation Accuracy: {training_results['validation_accuracy']:.4f}")
        print(f"Validation F1 Score: {training_results['validation_f1']:.4f}")
        
        # Make predictions
        print("\n=== MAKING PREDICTIONS ===")
        predictions = model.predict('cumulative_2025.10.04_07.24.46.csv', input_type='file')
        
        print(f"Predictions completed for {len(predictions)} objects")
        
        # Show sample predictions with detailed logging
        print("\nSample predictions:")
        for i in range(min(5, len(predictions))):
            pred = predictions[i]
            print(f"{pred['exoplanet_name']}: {pred['predicted_class']} (confidence: {pred['confidence']:.3f})")
            # Log detailed probabilities
            for class_name, prob in pred['class_probabilities'].items():
                logging.info(f"  {pred['exoplanet_name']} - {class_name}: {prob:.4f}")
        
        # Create and save plots
        print("\n=== CREATING PLOTS ===")
        freq_fig = model.plot_prediction_frequency(predictions)
        importance_fig = model.plot_feature_importance()
        
        # Save model
        model.save_model()
        
        # Generate comprehensive output summary
        print("\n=== OUTPUT SUMMARY ===")
        print(f"All outputs saved to: {model.output_dir}")
        print(f"  📁 Logs: {os.path.join(model.output_dir, 'logs')}")
        print(f"  📁 Predictions: {os.path.join(model.output_dir, 'predictions')}")
        print(f"  📁 Plots: {os.path.join(model.output_dir, 'plots')}")
        
        # Model info for UI
        info = model.get_model_info()
        print(f"\n=== MODEL INFO FOR UI ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Demonstrate new UI-friendly methods
        print("\n=== UI-FRIENDLY METHODS DEMO ===")
        
        # Method 1: Simple predictions (no file saving)
        simple_predictions = model.predict_simple('cumulative_2025.10.04_07.24.46.csv')
        print(f"✅ predict_simple(): Returns {len(simple_predictions)} predictions in memory")
        
        # Method 2: Organized UI format
        ui_format = model.predict_with_simple_format('cumulative_2025.10.04_07.24.46.csv')
        print(f"✅ predict_with_simple_format(): Organized data for UI")
        print(f"   Total predictions: {ui_format['total_predictions']}")
        print(f"   High confidence (>80%): {ui_format['high_confidence_count']}")
        print(f"   Class breakdown: {ui_format['class_summary']}")
        
        # Method 3: Simple lists
        names, classes, confidences, probs = model.get_predictions_as_lists('cumulative_2025.10.04_07.24.46.csv')
        print(f"✅ get_predictions_as_lists(): Returns 4 simple lists")
        print(f"   Names: {len(names)} items")
        print(f"   Classes: {len(classes)} items") 
        print(f"   Confidences: {len(confidences)} items")
        print(f"   Probabilities: {len(probs)} items")
        
        # Show sample of simple format
        print(f"\nSample UI data format:")
        for i in range(min(3, len(names))):
            print(f"  {names[i]}: {classes[i]} ({confidences[i]:.3f})")
        
        # Create UI-ready summary
        ui_summary = {
            "model_ready": True,
            "predictions_count": len(predictions),
            "output_directory": model.output_dir,
            "latest_predictions": predictions[:10],  # First 10 for preview
            "class_distribution": {
                pred['predicted_class']: sum(1 for p in predictions if p['predicted_class'] == pred['predicted_class'])
                for pred in predictions
            }
        }
        
        # Save UI summary
        ui_summary_path = os.path.join(model.output_dir, "ui_summary.json")
        with open(ui_summary_path, 'w') as f:
            json.dump(ui_summary, f, indent=2, default=str)
        
        print(f"\nUI summary saved to: {ui_summary_path}")
        print("Final model setup completed successfully!")
        
        return model, predictions, training_results
        
    except Exception as e:
        logging.error(f"Error in final model pipeline: {e}")
        raise e

if __name__ == "__main__":
    main()
