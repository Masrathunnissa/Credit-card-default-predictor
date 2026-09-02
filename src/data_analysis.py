"""
Data Analysis and Visualization Module
Analyzes statistical measures, duplicates, unique values, and correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime


def analyze_statistical_measures(df):
    """
    Generate statistical summary of the dataframe.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Statistical summary and HTML representation
    """
    try:
        stats_summary = df.describe().T
        stats_html = stats_summary.to_html(classes='table table-striped')
        
        print("✅ Statistical measures calculated")
        return {
            'stats': stats_summary,
            'html': stats_html,
            'count': len(df),
            'columns': len(df.columns)
        }
    except Exception as e:
        print(f"❌ Error in statistical analysis: {e}")
        return None


def check_duplicates(df):
    """
    Check for duplicate rows in the dataframe.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Duplicate count and percentage
    """
    try:
        duplicate_count = df.duplicated().sum()
        total_rows = len(df)
        duplicate_percentage = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"✅ Duplicate check: {duplicate_count} duplicates found ({duplicate_percentage:.2f}%)")
        return {
            'count': duplicate_count,
            'percentage': duplicate_percentage,
            'total_rows': total_rows
        }
    except Exception as e:
        print(f"❌ Error checking duplicates: {e}")
        return None


def check_unique_values(df):
    """
    Check unique values in each column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Unique value counts for each column
    """
    try:
        unique_counts = df.nunique()
        unique_dict = unique_counts.to_dict()
        
        unique_df = pd.DataFrame({
            'Column': unique_counts.index,
            'Unique Values': unique_counts.values,
            'Total Rows': len(df),
            'Percentage': (unique_counts.values / len(df) * 100).round(2)
        })
        
        unique_html = unique_df.to_html(classes='table table-striped', index=False)
        
        print(f"✅ Unique values checked for {len(unique_dict)} columns")
        return {
            'unique_dict': unique_dict,
            'html': unique_html,
            'total_columns': len(unique_dict)
        }
    except Exception as e:
        print(f"❌ Error checking unique values: {e}")
        return None


def generate_correlation_heatmap(df, output_path=None):
    """
    Generate correlation heatmap visualization.
    
    Args:
        df: Input DataFrame
        output_path: Path to save the heatmap image
    
    Returns:
        dict: Base64 encoded image and correlation matrix
    """
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            print("⚠️ No numeric columns found for correlation")
            return None
        
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Heatmap - Feature Relationships", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to file if path provided
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"✅ Heatmap saved to: {output_path}")
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return {
            'image_base64': f'data:image/png;base64,{img_base64}',
            'correlation_matrix': correlation_matrix.to_html(classes='table table-striped'),
            'strong_correlations': find_strong_correlations(correlation_matrix)
        }
    except Exception as e:
        print(f"❌ Error generating heatmap: {e}")
        return None


def find_strong_correlations(corr_matrix, threshold=0.7):
    """
    Find strong correlations in the correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (default 0.7)
    
    Returns:
        list: List of strong correlation pairs
    """
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_corrs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': round(corr_value, 3)
                })
    
    return strong_corrs


def analyze_payment_history(df, target_column='default.payment.next.month'):
    """
    Analyze payment history columns vs default target.
    
    Args:
        df: Input DataFrame
        target_column: Target column name for default
    
    Returns:
        dict: Analysis results with base64 encoded images
    """
    try:
        payment_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        images = []
        
        # Filter columns that exist in the dataframe
        existing_payment_cols = [col for col in payment_columns if col in df.columns]
        
        if not existing_payment_cols:
            print("⚠️ No payment history columns found")
            return None
        
        # Check if target column exists
        if target_column not in df.columns:
            # Try alternative names
            possible_targets = [col for col in df.columns if 'default' in col.lower()]
            if possible_targets:
                target_column = possible_targets[0]
            else:
                print("⚠️ Target column not found")
                return None
        
        for col in existing_payment_cols:
            try:
                plt.figure(figsize=(10, 6))
                
                # Handle categorical data
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    sns.countplot(
                        x=col,
                        hue=target_column,
                        data=df,
                        palette="Set2"
                    )
                else:
                    # For continuous data, use histogram
                    df.groupby(target_column)[col].plot(kind='hist', legend=True)
                
                plt.title(f'Payment Status: {col} vs Default', fontsize=12, fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Convert to base64
                img = io.BytesIO()
                plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
                img.seek(0)
                img_base64 = base64.b64encode(img.getvalue()).decode()
                
                images.append({
                    'column': col,
                    'image': f'data:image/png;base64,{img_base64}'
                })
                
                plt.close()
            except Exception as e:
                print(f"⚠️ Error plotting {col}: {e}")
                continue
        
        print(f"✅ Payment history analysis completed for {len(images)} columns")
        return {
            'images': images,
            'columns_analyzed': len(images)
        }
    except Exception as e:
        print(f"❌ Error in payment history analysis: {e}")
        return None


def generate_data_quality_report(df):
    """
    Generate comprehensive data quality report.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Data quality metrics
    """
    try:
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
            'data_types': df.dtypes.to_dict()
        }
        
        print("✅ Data quality report generated")
        return quality_report
    except Exception as e:
        print(f"❌ Error generating quality report: {e}")
        return None
