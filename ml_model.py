"""
Machine Learning Model for Exoplanet Classification
Multi-class classification to predict exoplanet disposition (CONFIRMED, FALSE POSITIVE, CANDIDATE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network models will be skipped.")

from data_processing import process_data

class ExoplanetClassifier:
    """Multi-class classifier for exoplanet disposition prediction."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names = None
        self.predictions = None
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for ML training."""
        logging.info("Loading and preparing data...")
        
        # Get processed data
        df = process_data(save_csv=True, setup_logging=False)
        
        # Check if target column exists
        if 'koi_disposition' not in df.columns:
            raise ValueError("Target column 'koi_disposition' not found in processed data")
        
        # Separate features and target
        X = df.drop('koi_disposition', axis=1)
        y = df['koi_disposition']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle any remaining categorical variables in features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        logging.info(f"Data shape: {X.shape}")
        logging.info(f"Target classes: {self.label_encoder.classes_}")
        logging.info(f"Class distribution: {pd.Series(y).value_counts()}")
        
        return X, pd.Series(y_encoded)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models and compare performance."""
        logging.info("Training multiple models...")
        
        # Define models to train
        models_config = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        model_scores = {}
        
        for name, model in models_config.items():
            logging.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            
            # Train on full training set
            model.fit(X_train, y_train)
            self.models[name] = model
            
            model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            logging.info(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model_scores
    
    def create_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create a neural network architecture for exoplanet classification."""
        model = keras.Sequential([
            # Input layer with dropout for regularization
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers with decreasing size
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train neural network with proper validation."""
        if not TENSORFLOW_AVAILABLE:
            logging.warning("TensorFlow not available. Skipping neural network training.")
            return {}
        
        logging.info("Training Neural Network...")
        
        # Convert labels to categorical
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Create model
        model = self.create_neural_network(X_train.shape[1], num_classes)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store the trained model
        self.models['Neural Network'] = model
        
        # Calculate validation scores
        val_predictions = model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_f1 = f1_score(y_val, val_pred_classes, average='weighted')
        val_accuracy = accuracy_score(y_val, val_pred_classes)
        
        logging.info(f"Neural Network - Validation F1 Score: {val_f1:.4f}")
        logging.info(f"Neural Network - Validation Accuracy: {val_accuracy:.4f}")
        
        return {
            'model': model,
            'history': history,
            'val_f1': val_f1,
            'val_accuracy': val_accuracy
        }
    
    def plot_neural_network_training(self, history) -> None:
        """Plot neural network training history."""
        if history is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('neural_network_training.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Perform hyperparameter tuning on the best performing model."""
        logging.info("Performing hyperparameter tuning...")
        
        # Use Random Forest for hyperparameter tuning (typically performs well)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on test set."""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'predictions': y_pred
        }
    
    def plot_results(self, y_test: pd.Series, y_pred: np.ndarray) -> None:
        """Plot confusion matrix and feature importance."""
        plt.figure(figsize=(15, 5))
        
        # Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(1, 3, 2)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            sns.barplot(data=feature_importance, y='feature', x='importance')
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
        
        # Class Distribution
        plt.subplot(1, 3, 3)
        class_counts = pd.Series(y_test).value_counts()
        plt.pie(class_counts.values, labels=self.label_encoder.classes_, autopct='%1.1f%%')
        plt.title('Test Set Class Distribution')
        
        plt.tight_layout()
        plt.savefig('ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path: str = 'exoplanet_classifier.joblib') -> None:
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        logging.info(f"Model saved to {model_path}")
    
    def predict_new_data(self, X_new: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        predictions_encoded = self.best_model.predict(X_new)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        self.predictions = predictions

        return predictions

def main():
    """Main training pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize classifier
    classifier = ExoplanetClassifier()
    
    try:
        # Load and prepare data
        X, y = classifier.load_and_prepare_data()
        
        # Split data (train/validation/test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_val_scaled = classifier.scaler.transform(X_val)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Train traditional ML models
        model_scores = classifier.train_models(X_train_scaled, y_train)
        @property
        def predictions(self):
            return self.predictions
        # Train neural network
        if TENSORFLOW_AVAILABLE:
            nn_results = classifier.train_neural_network(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            if nn_results:
                classifier.plot_neural_network_training(nn_results.get('history'))
        
        # Hyperparameter tuning on best traditional model
        classifier.hyperparameter_tuning(X_train_scaled, y_train)
        
        # Compare all models on test set
        print("\n=== MODEL COMPARISON ===")
        best_score = 0
        best_model_name = ""
        
        for model_name, model in classifier.models.items():
            if model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
                # Handle neural network predictions
                test_predictions = model.predict(X_test_scaled)
                test_pred_classes = np.argmax(test_predictions, axis=1)
                test_f1 = f1_score(y_test, test_pred_classes, average='weighted')
                test_accuracy = accuracy_score(y_test, test_pred_classes)
            else:
                # Handle sklearn models
                results = classifier.evaluate_model(model, X_test_scaled, y_test)
                test_f1 = results['f1_score']
                test_accuracy = results['accuracy']
                test_pred_classes = results['predictions']
            
            print(f"{model_name}:")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test F1 Score: {test_f1:.4f}")
            
            if test_f1 > best_score:
                best_score = test_f1
                best_model_name = model_name
                classifier.best_model = model
        
        print(f"\nBest Model: {best_model_name} (F1 Score: {best_score:.4f})")
        
        # Detailed evaluation of best model
        if best_model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
            predictions = np.argmax(classifier.best_model.predict(X_test_scaled), axis=1)
        else:
            predictions = classifier.best_model.predict(X_test_scaled)
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions, 
                                  target_names=classifier.label_encoder.classes_))
        
        # Plot results
        classifier.plot_results(y_test, predictions)
        
        # Save model
        classifier.save_model()
        
        logging.info("ML pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in ML pipeline: {e}")
        raise e

if __name__ == "__main__":
    main()