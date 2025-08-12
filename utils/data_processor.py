import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file into DataFrame"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def clean_numeric_column(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Clean and convert column to numeric"""
        return pd.to_numeric(df[column].str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    def filter_data(self, df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Filter DataFrame based on conditions"""
        filtered_df = df.copy()
        
        for column, condition in conditions.items():
            if isinstance(condition, dict):
                if 'min' in condition:
                    filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                if 'max' in condition:
                    filtered_df = filtered_df[filtered_df[column] <= condition['max']]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]
        
        return filtered_df
    
    def calculate_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Calculate correlation between two columns"""
        return df[col1].corr(df[col2])
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for DataFrame"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'summary': df.describe().to_dict()
        }
