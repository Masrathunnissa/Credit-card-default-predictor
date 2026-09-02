"""
Feature Importance Analysis Module
Calculates and visualizes feature importance from trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime


def calculate_feature_importance_from_model(model, feature_names):
    """
    Calculate feature importance from a trained model.
    
    Args:
        model: Trained sklearn model (must have coef_ or feature_importances_)
        feature_names: List of feature names
    
    Returns:
        dict: Feature importance data and visualization
    """
    try:
        # Try to get coefficients (for linear models)
        if hasattr(model, 'coef_'):
            feature_importance = pd.Series(
                model.coef_[0],
                index=feature_names
            )
            importance_type = "Model Coefficients"
        # Try to get feature importances (for tree-based models)
        elif hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            )
            importance_type = "Tree-based Feature Importance"
        else:
            print("⚠️ Model does not have feature importance/coefficients")
            return None
        
        # Sort by absolute value
        feature_importance_sorted = feature_importance.abs().sort_values(ascending=False)
        
        print(f"✅ Feature importance calculated ({importance_type})")
        print(f"   Top 5 features: {', '.join(feature_importance_sorted.head(5).index)}")
        
        return {
            'feature_importance': feature_importance,
            'feature_importance_sorted': feature_importance_sorted,
            'importance_type': importance_type,
            'top_features': feature_importance_sorted.head(10).to_dict()
        }
    except Exception as e:
        print(f"❌ Error calculating feature importance: {e}")
        return None


def plot_feature_importance(feature_importance_dict, output_path=None, top_n=15):
    """
    Create visualization of feature importance.
    
    Args:
        feature_importance_dict: Dictionary from calculate_feature_importance_from_model
        output_path: Path to save the plot
        top_n: Number of top features to display
    
    Returns:
        dict: Base64 encoded image and statistics
    """
    try:
        if not feature_importance_dict:
            return None
        
        sorted_importance = feature_importance_dict['feature_importance_sorted']
        
        # Display top N features
        top_features = sorted_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features.values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score (Absolute Value)', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Important Features\n({feature_importance_dict["importance_type"]})', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Color bars based on importance
        for i, (feature, value) in enumerate(top_features.items()):
            bars[i].set_color(plt.cm.RdYlGn(value / top_features.max()))
        
        plt.tight_layout()
        
        # Save to file if path provided
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to: {output_path}")
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Create HTML table of all features
        importance_df = pd.DataFrame({
            'Feature': sorted_importance.index,
            'Importance': sorted_importance.values.round(4),
            'Rank': range(1, len(sorted_importance) + 1)
        })
        
        importance_html = importance_df.to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return {
            'image_base64': f'data:image/png;base64,{img_base64}',
            'html_table': importance_html,
            'top_features': top_features.to_dict(),
            'total_features': len(sorted_importance),
            'top_n': top_n
        }
    except Exception as e:
        print(f"❌ Error plotting feature importance: {e}")
        return None


def generate_feature_importance_report(feature_importance_dict):
    """
    Generate detailed feature importance report.
    
    Args:
        feature_importance_dict: Dictionary from calculate_feature_importance_from_model
    
    Returns:
        dict: Detailed report with statistics
    """
    try:
        if not feature_importance_dict:
            return None
        
        sorted_importance = feature_importance_dict['feature_importance_sorted']
        
        # Calculate statistics
        stats = {
            'total_features': len(sorted_importance),
            'max_importance': round(sorted_importance.max(), 4),
            'min_importance': round(sorted_importance.min(), 4),
            'mean_importance': round(sorted_importance.mean(), 4),
            'std_importance': round(sorted_importance.std(), 4),
            'top_1_feature': sorted_importance.index[0],
            'top_1_importance': round(sorted_importance.iloc[0], 4),
            'cumulative_importance_top_5': round(sorted_importance.head(5).sum(), 4),
            'cumulative_importance_top_10': round(sorted_importance.head(10).sum(), 4),
            'importance_type': feature_importance_dict['importance_type']
        }
        
        # Determine feature categories by importance
        percentile_75 = sorted_importance.quantile(0.75)
        percentile_25 = sorted_importance.quantile(0.25)
        
        high_importance = sorted_importance[sorted_importance >= percentile_75].index.tolist()
        medium_importance = sorted_importance[(sorted_importance < percentile_75) & 
                                             (sorted_importance >= percentile_25)].index.tolist()
        low_importance = sorted_importance[sorted_importance < percentile_25].index.tolist()
        
        stats['high_importance_features'] = high_importance
        stats['medium_importance_features'] = medium_importance
        stats['low_importance_features'] = low_importance
        
        print(f"✅ Feature importance report generated")
        print(f"   High importance: {len(high_importance)} features")
        print(f"   Medium importance: {len(medium_importance)} features")
        print(f"   Low importance: {len(low_importance)} features")
        
        return stats
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return None


def identify_top_predictive_features(feature_importance_dict, threshold=0.8):
    """
    Identify features that contribute to 80% of total importance.
    
    Args:
        feature_importance_dict: Dictionary from calculate_feature_importance_from_model
        threshold: Cumulative importance threshold (default 0.8 = 80%)
    
    Returns:
        dict: Features and their contribution
    """
    try:
        if not feature_importance_dict:
            return None
        
        sorted_importance = feature_importance_dict['feature_importance_sorted']
        total_importance = sorted_importance.sum()
        
        # Calculate cumulative importance
        cumulative_importance = 0
        important_features = []
        
        for feature, importance in sorted_importance.items():
            cumulative_importance += importance
            important_features.append({
                'feature': feature,
                'importance': round(importance, 4),
                'cumulative_importance': round(cumulative_importance, 4),
                'cumulative_percentage': round((cumulative_importance / total_importance * 100), 2)
            })
            
            if (cumulative_importance / total_importance) >= threshold:
                break
        
        print(f"✅ Identified {len(important_features)} features contributing to {threshold*100:.0f}% importance")
        
        return {
            'features': important_features,
            'count': len(important_features),
            'cumulative_importance': round(cumulative_importance, 4),
            'total_importance': round(total_importance, 4)
        }
    except Exception as e:
        print(f"❌ Error identifying top features: {e}")
        return None
