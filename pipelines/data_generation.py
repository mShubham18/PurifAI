import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

def generate_synthetic_data(df, num_points):
    """
    Generate synthetic data points based on the input dataset using statistical methods.
    
    Args:
        df (pd.DataFrame): Input dataset
        num_points (int): Number of synthetic points to generate
        
    Returns:
        pd.DataFrame: Generated synthetic dataset
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create a copy of the original dataframe
        synthetic_df = pd.DataFrame(columns=df.columns)
        
        # Generate data for each column
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numeric columns, use normal distribution
                mean = df[column].mean()
                std = df[column].std()
                synthetic_df[column] = np.random.normal(mean, std, num_points)
                
                # Ensure values stay within original range
                min_val = df[column].min()
                max_val = df[column].max()
                synthetic_df[column] = synthetic_df[column].clip(min_val, max_val)
                
            elif pd.api.types.is_string_dtype(df[column]):
                # For categorical columns, sample from existing values
                unique_values = df[column].unique()
                synthetic_df[column] = np.random.choice(unique_values, num_points)
                
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                # For datetime columns, generate dates within the original range
                min_date = df[column].min()
                max_date = df[column].max()
                date_range = (max_date - min_date).days
                random_days = np.random.randint(0, date_range, num_points)
                synthetic_df[column] = min_date + pd.to_timedelta(random_days, unit='D')
                
            else:
                # For other types, try to maintain the original type
                synthetic_df[column] = df[column].sample(n=num_points, replace=True).values
        
        # Ensure data types match original dataset
        for column in df.columns:
            synthetic_df[column] = synthetic_df[column].astype(df[column].dtype)
        
        return synthetic_df
            
    except Exception as e:
        raise Exception(f"Data generation failed: {str(e)}") 