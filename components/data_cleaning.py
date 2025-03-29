import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

def remove_special_chars(text):
    """Removes special symbols, emojis, and unnecessary characters from text."""
    if isinstance(text, str):
        return re.sub(r"[^\w\s]", "", text)  # Keeps only letters, numbers, and spaces
    return text

def clean_and_standardize_data(file_path, corr_threshold=0.9):
    """
    Cleans messy dataset: fixes datatypes, fills missing values, removes outliers,
    drops highly correlated columns, removes special symbols, and standardizes numerical features.
    """
    
    # Load dataset
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Drop duplicates & completely empty columns
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')

    # Convert column names to lowercase & remove leading/trailing spaces
    df.columns = df.columns.str.lower().str.strip()

    # Detect and fix incorrect data types
    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if str(x).strip().lower() in ["nan", "none", "null", "n/a", ""] else x)

        if df[col].dtype == 'object':  # If it's a string column
            try:
                df[col] = pd.to_numeric(df[col])  # Try converting to numeric
            except:
                pass  # If it fails, keep it as a categorical/text column

    # Remove special symbols & emojis from text columns
    text_cols = df.select_dtypes(include=['object']).columns
    df[text_cols] = df[text_cols].applymap(remove_special_chars)

    # Fill missing values intelligently
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Numeric columns
            df[col] = df[col].fillna(df[col].median())  # Use median for robustness
        else:  # Categorical columns
            df[col] = df[col].fillna(df[col].mode()[0])  # Use most common value

    # Handle outliers using IQR method
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Drop highly correlated features
    corr_matrix = df[numeric_cols].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    df = df.drop(columns=drop_cols)

    # ✅ Recalculate numeric_cols after dropping correlated features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Standardize numerical columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


"""# Example usage
file_path = "dataset.csv"
cleaned_standardized_df = clean_and_standardize_data(file_path)

print(cleaned_standardized_df.head())
"""