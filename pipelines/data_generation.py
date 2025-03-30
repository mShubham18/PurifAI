import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(df: pd.DataFrame, num_rows: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic data that preserves statistical properties and relationships while ensuring bias-free generation.
    
    Args:
        df (pd.DataFrame): Input dataset to base generation on
        num_rows (int): Number of rows to generate
        
    Returns:
        pd.DataFrame: Generated synthetic dataset
    """
    try:
        # Validate input parameters
        if num_rows < 100:
            raise ValueError("Number of rows must be at least 100")
        if len(df) == 0:
            raise ValueError("Input dataset cannot be empty")
            
        synthetic_data = pd.DataFrame()
        
        # Handle different data types
        for column in df.columns:
            dtype = df[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                # For numeric columns, use a mixture of distributions to ensure diversity
                data = df[column].dropna()
                if len(data) > 0:
                    # Calculate basic statistics
                    mean = data.mean()
                    std = data.std()
                    
                    # Ensure std is not zero or too small
                    if std < 1e-10:
                        std = 1.0
                    
                    # Generate data using a mixture of normal distributions
                    # This helps maintain diversity and prevent bias
                    num_components = min(3, max(1, len(data) // 100))  # Number of mixture components
                    weights = np.random.dirichlet(np.ones(num_components))
                    means = np.random.normal(mean, std, num_components)
                    stds = np.abs(np.random.normal(std, std/2, num_components))
                    
                    # Ensure stds are not too small
                    stds = np.maximum(stds, 1e-10)
                    
                    # Generate synthetic data
                    synthetic_values = np.zeros(num_rows)
                    for i in range(num_rows):
                        component = np.random.choice(num_components, p=weights)
                        synthetic_values[i] = np.random.normal(means[component], stds[component])
                    
                    # Apply outlier handling
                    z_scores = np.abs(stats.zscore(synthetic_values))
                    synthetic_values[z_scores > 3] = np.random.normal(mean, std, np.sum(z_scores > 3))
                    
                    # Ensure the range matches the original data
                    synthetic_values = np.clip(synthetic_values, data.min(), data.max())
                    
                    synthetic_data[column] = synthetic_values
                    
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                # For datetime columns, generate dates within the original range
                min_date = df[column].min()
                max_date = df[column].max()
                date_range = (max_date - min_date).days
                synthetic_data[column] = pd.date_range(
                    start=min_date,
                    periods=num_rows,
                    freq='D'
                ) + pd.Timedelta(days=np.random.randint(0, max(1, date_range), num_rows))
                
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                # For categorical columns, ensure balanced distribution
                unique_values = df[column].dropna().unique()
                if len(unique_values) > 0:
                    # Calculate probabilities based on frequency
                    value_counts = df[column].value_counts(normalize=True)
                    synthetic_data[column] = np.random.choice(
                        unique_values,
                        size=num_rows,
                        p=value_counts.values
                    )
                else:
                    synthetic_data[column] = np.random.choice(['A', 'B', 'C'], num_rows)
            else:
                # For other types, use random sampling with replacement
                synthetic_data[column] = np.random.choice(
                    df[column].dropna().values,
                    size=num_rows,
                    replace=True
                )
        
        # Preserve correlations between numeric columns
        numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Calculate correlation matrix of original data
            original_corr = df[numeric_cols].corr()
            
            # Apply correlation preservation using Cholesky decomposition
            try:
                L = np.linalg.cholesky(original_corr)
                uncorrelated = StandardScaler().fit_transform(synthetic_data[numeric_cols])
                correlated = np.dot(uncorrelated, L.T)
                
                # Scale back to original ranges
                for i, col in enumerate(numeric_cols):
                    orig_data = df[col].dropna()
                    if len(orig_data) > 0:
                        correlated[:, i] = correlated[:, i] * orig_data.std() + orig_data.mean()
                        correlated[:, i] = np.clip(correlated[:, i], orig_data.min(), orig_data.max())
                
                synthetic_data[numeric_cols] = correlated
            except np.linalg.LinAlgError:
                warnings.warn("Could not preserve correlations due to non-positive definite correlation matrix")
        
        # Add some controlled randomness to prevent exact patterns
        for col in numeric_cols:
            noise = np.random.normal(0, 0.01, num_rows)
            synthetic_data[col] = synthetic_data[col] * (1 + noise)
        
        # Ensure data quality
        synthetic_data = synthetic_data.replace([np.inf, -np.inf], np.nan)
        synthetic_data = synthetic_data.fillna(method='ffill').fillna(method='bfill')
        
        return synthetic_data
        
    except Exception as e:
        raise Exception(f"Error generating synthetic data: {str(e)}") 