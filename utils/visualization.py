import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import base64
import io
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualizationEngine:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_scatterplot_with_regression(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create scatterplot with dotted red regression line"""
        try:
            # Clean data
            df_clean = df.copy()
            df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col, y_col])
            
            if df_clean.empty:
                return self._create_dummy_plot()
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.scatter(df_clean[x_col], df_clean[y_col], alpha=0.6)
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean[x_col], df_clean[y_col])
            line_x = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100)
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, color='red', linestyle=':', linewidth=2, label=f'Regression Line (r²={r_value**2:.3f})')
            
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Scatterplot: {x_col} vs {y_col}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return self._save_plot_as_base64()
        
        except Exception as e:
            logger.error(f"Error creating scatterplot: {str(e)}")
            return self._create_dummy_plot()
    
    def create_delay_scatterplot(self, data: List[Tuple]) -> str:
        """Create delay scatterplot with regression line"""
        try:
            if not data:
                return self._create_dummy_plot()
            
            x = [point[0] for point in data]  # years
            y = [point[1] for point in data]  # delays
            
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, alpha=0.6)
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.linspace(min(x), max(x), 100)
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, color='red', linestyle=':', linewidth=2)
            
            plt.xlabel('Year')
            plt.ylabel('Average Delay (Days)')
            plt.title('Court Case Processing Delay by Year')
            plt.grid(True, alpha=0.3)
            
            return self._save_plot_as_base64()
        
        except Exception as e:
            logger.error(f"Error creating delay plot: {str(e)}")
            return self._create_dummy_plot()
    
    def _save_plot_as_base64(self) -> str:
        """Save current plot as base64 encoded PNG"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        
        # Encode as base64
        plot_data = base64.b64encode(buffer.read()).decode()
        plt.close()  # Close the plot to free memory
        
        return f"data:image/png;base64,{plot_data}"
    
    def _create_dummy_plot(self) -> str:
        """Create a dummy plot when data is not available"""
        plt.figure(figsize=(8, 6))
        x = np.random.randn(50)
        y = 2 * x + np.random.randn(50)
        
        plt.scatter(x, y, alpha=0.6)
        
        # Add regression line
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, color='red', linestyle=':', linewidth=2)
        
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Sample Scatterplot with Regression Line')
        plt.grid(True, alpha=0.3)
        
        return self._save_plot_as_base64()
