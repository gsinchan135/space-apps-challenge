"""
Data processing script 
Input: raw csv file
Output: cleaned csv file for machine leanring model

Key Functions:
1. load_data: Load raw data from csv file
2. remove_leakage_cols: Remove columns that may cause data leakage
3. handle_missing_values: Fill or drop missing values
4. encode_categorical_vars (optional): Convert categorical variables to numerical
5. remove_outliers: Remove outliers from numerical columns
6. standardize_numerical_vars: Standardize numerical variables
7. drop irrelevant_cols: Drop irrelevant columns
8. save_cleaned_data: Save cleaned data to csv file

feature engineering future additions
- use quarters as a feature
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import logging
from typing import List, Optional
import argparse
import sys

# Load Data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path, comment='#')
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

# Remove Leakage Columns
def remove_leakage_cols(df: pd.DataFrame, leakage_cols: List[str]) -> pd.DataFrame:
    """Remove columns that may cause data leakage."""
    df = df.drop(columns=leakage_cols, errors='ignore')
    logging.info(f"Removed leakage columns: {leakage_cols}")
    return df

# Handle Missing Values
def handle_missing_values(df: pd.DataFrame, missing_threshold: float = 0.2) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        missing_threshold: Threshold for dropping columns (default: 0.2 = 20%)
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    original_columns = df_copy.columns.tolist()
    
    # Calculate missing percentage for each column
    missing_percentages = df_copy.isnull().sum() / len(df_copy)
    
    # Identify columns to drop (>20% missing values)
    cols_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()
    
    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        logging.info(f"Dropped columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
    
    # Handle remaining missing values
    for column in df_copy.columns:
        if df_copy[column].isnull().sum() > 0:
            if df_copy[column].dtype in ['float64', 'int64', 'float32', 'int32']:
                # Median imputation for numerical columns
                median_value = df_copy[column].median()
                df_copy[column].fillna(median_value, inplace=True)
                logging.info(f"Applied median imputation to column '{column}' (median: {median_value})")
            else:
                # Mode imputation for categorical columns
                mode_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else 'Unknown'
                df_copy[column].fillna(mode_value, inplace=True)
                logging.info(f"Applied mode imputation to column '{column}' (mode: {mode_value})")
    
    # Log summary
    remaining_missing = df_copy.isnull().sum().sum()
    logging.info(f"Missing value handling complete. Remaining missing values: {remaining_missing}")
    
    return df_copy


# Drop text categorical variables
def drop_text_categorical_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop text categorical variables from the DataFrame, but keep koi_disposition."""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Keep koi_disposition (likely target variable)
    cols_to_keep = ['koi_disposition']
    text_cols_to_drop = [col for col in text_cols if col not in cols_to_keep]
    
    if text_cols_to_drop:
        df = df.drop(columns=text_cols_to_drop, errors='ignore')
        logging.info(f"Dropped text categorical variables: {text_cols_to_drop}")
        logging.info(f"Kept text columns: {[col for col in cols_to_keep if col in df.columns]}")
    else:
        logging.info("No text categorical variables to drop")
    
    return df

# Remove error columns
def remove_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that end with 'err1' or 'err2'."""
    df_copy = df.copy()
    
    # Find columns ending with 'err1' or 'err2'
    error_cols = [col for col in df_copy.columns if col.endswith('err1') or col.endswith('err2')]
    
    if error_cols:
        df_copy = df_copy.drop(columns=error_cols)
        logging.info(f"Removed error columns: {error_cols}")
    else:
        logging.info("No error columns (ending with 'err1' or 'err2') found to remove")
    
    return df_copy


# Remove outliers
def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using Z-score method."""
    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
        df_copy = df_copy[z_scores < z_threshold]
    
    logging.info(f"Removed outliers using Z-score threshold: {z_threshold}")
    return df_copy

# Standardize numerical variables
def standardize_numerical_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numerical variables using StandardScaler."""
    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
        logging.info(f"Standardized numerical columns: {numerical_cols.tolist()}")
    
    return df_copy

# Save cleaned data
def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned data to CSV file."""
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to: {output_path}")

def process_data(input_file: str = 'cumulative_2025.10.04_07.24.46.csv', 
                output_file: Optional[str] = 'cleaned_data.csv',
                save_csv: bool = True,
                setup_logging: bool = True) -> pd.DataFrame:
    """
    Main data processing pipeline that returns cleaned DataFrame.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save cleaned CSV (if save_csv=True)
        save_csv: Whether to save the cleaned data to CSV
        setup_logging: Whether to setup logging (set False if already configured)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for ML
    """
    # Setup logging if requested
    if setup_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
    
    # Define columns that might cause data leakage (adjust based on your specific use case)
    leakage_columns = [
        "koi_pdisposition",  # Robovetter's label (automated classification)
        "koi_vet_stat",      # Human vetting status
        "koi_vet_date",      # Date vetting was done
        "koi_comment",       # Human notes about classification
        "koi_disp_prov",     # Provenance of disposition (manual vs automated)
        "koi_score",         # Robovetter confidence score
        "kepler_name",       # Only assigned to confirmed planets
        "kepid",             # Unique ID, not predictive
        "kepoi_name"         # Unique KOI identifier
    ]  

    try:
        # Step 1: Load data
        logging.info("Starting data processing pipeline...")
        df = load_data(input_file)
        logging.info(f"Initial data shape: {df.shape}")
        
        # Step 2: Remove leakage columns
        df = remove_leakage_cols(df, leakage_columns)
        
        # Step 3: Handle missing values
        df = handle_missing_values(df, missing_threshold=0.2)
        logging.info(f"Data shape after handling missing values: {df.shape}")
        
        # Step 4: Drop text categorical variables
        df = drop_text_categorical_vars(df)
        logging.info(f"Data shape after dropping text columns: {df.shape}")
        
        # Step 5: Remove error columns (ending with err1 or err2)
        df = remove_error_columns(df)
        logging.info(f"Data shape after removing error columns: {df.shape}")
        
        # Step 6: Remove outliers
        # df = remove_outliers(df, z_threshold=3.0)
        # logging.info(f"Data shape after removing outliers: {df.shape}")
        
        # Step 7: Standardize numerical variables
        # df = standardize_numerical_vars(df)
        
        # Step 8: Save cleaned data (optional)
        if save_csv and output_file:
            save_cleaned_data(df, output_file)
        
        logging.info("Data processing pipeline completed successfully!")
        logging.info(f"Final data shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in data processing pipeline: {e}")
        raise e

def main():
    """Command line interface for data processing."""
    # Process data and display summary
    df = process_data()
    
    # Display summary statistics
    print("\n=== DATA PROCESSING SUMMARY ===")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows of cleaned data:")
    print(df.head())

