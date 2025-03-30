from components.model_configuration import model_config
import pandas as pd
import numpy as np
from datetime import datetime
import re

def clean_text(text):
    """Clean text to ensure ASCII compatibility"""
    # Replace common Unicode arrows with ASCII alternatives
    text = text.replace('↓', '->')
    text = text.replace('↑', '<-')
    text = text.replace('→', '->')
    text = text.replace('←', '<-')
    text = text.replace('⇒', '=>')
    text = text.replace('⇐', '<=')
    
    # Remove any other non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def generate_report(df):
    model = model_config()
    
    # Basic statistics
    num_rows, num_cols = df.shape
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Generate report sections
    sections = []
    
    # 1. Executive Summary
    summary_prompt = f"""Create a concise executive summary (2-3 sentences) for this dataset:
    Number of rows: {num_rows}
    Number of columns: {num_cols}
    Numeric columns: {list(numeric_cols)}
    Categorical columns: {list(categorical_cols)}
    """
    summary = model.generate_content(summary_prompt).text
    sections.append(f"# Executive Summary\n\n{clean_text(summary)}\n\n")
    
    # 2. Dataset Overview
    sections.append("## Dataset Overview\n\n")
    sections.append(f"- Total Records: {num_rows:,}\n")
    sections.append(f"- Total Features: {num_cols:,}\n")
    sections.append(f"- Numeric Features: {len(numeric_cols):,}\n")
    sections.append(f"- Categorical Features: {len(categorical_cols):,}\n\n")
    
    # 3. Column Analysis
    sections.append("## Column Analysis\n\n")
    for col in df.columns:
        col_type = df[col].dtype
        missing_values = df[col].isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        if col_type in ['int64', 'float64']:
            stats = df[col].describe()
            sections.append(f"### {col} (Numeric)\n")
            sections.append(f"- Data Type: {col_type}\n")
            sections.append(f"- Missing Values: {missing_values:,} ({missing_percentage:.2f}%)\n")
            sections.append(f"- Mean: {stats['mean']:.2f}\n")
            sections.append(f"- Median: {stats['50%']:.2f}\n")
            sections.append(f"- Standard Deviation: {stats['std']:.2f}\n")
            sections.append(f"- Min: {stats['min']:.2f}\n")
            sections.append(f"- Max: {stats['max']:.2f}\n\n")
        else:
            unique_values = df[col].nunique()
            sections.append(f"### {col} (Categorical)\n")
            sections.append(f"- Data Type: {col_type}\n")
            sections.append(f"- Missing Values: {missing_values:,} ({missing_percentage:.2f}%)\n")
            sections.append(f"- Unique Values: {unique_values:,}\n")
            sections.append(f"- Most Common Value: {df[col].mode().iloc[0]}\n\n")
    
    # 4. Key Metrics and Insights
    metrics_prompt = f"""Analyze this dataset and provide key metrics and insights (3-4 bullet points):
    Dataset shape: {df.shape}
    Numeric columns: {list(numeric_cols)}
    Categorical columns: {list(categorical_cols)}
    """
    metrics = model.generate_content(metrics_prompt).text
    sections.append("## Key Metrics and Insights\n\n")
    sections.append(f"{clean_text(metrics)}\n\n")
    
    # 5. Data Distribution Analysis
    sections.append("## Data Distribution Analysis\n\n")
    for col in numeric_cols:
        sections.append(f"### {col}\n")
        sections.append(f"- Skewness: {df[col].skew():.2f}\n")
        sections.append(f"- Kurtosis: {df[col].kurtosis():.2f}\n")
        sections.append(f"- Quartiles:\n")
        sections.append(f"  - Q1: {df[col].quantile(0.25):.2f}\n")
        sections.append(f"  - Q2: {df[col].quantile(0.50):.2f}\n")
        sections.append(f"  - Q3: {df[col].quantile(0.75):.2f}\n\n")
    
    # 6. Recommendations
    recommendations_prompt = f"""Based on this dataset analysis, provide 3-4 actionable recommendations for data usage and potential improvements:
    Dataset shape: {df.shape}
    Numeric columns: {list(numeric_cols)}
    Categorical columns: {list(categorical_cols)}
    """
    recommendations = model.generate_content(recommendations_prompt).text
    sections.append("## Recommendations\n\n")
    sections.append(f"{clean_text(recommendations)}\n\n")
    
    # 7. Report Metadata
    sections.append("---\n")
    sections.append(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Combine all sections
    report = "\n".join(sections)
    
    return report