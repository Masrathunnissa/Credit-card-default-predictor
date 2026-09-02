"""
Model Comparison Module
Uses LazyClassifier to compare multiple classification models
"""

import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt


def compare_models_lazy_classifier(X_train, X_test, y_train, y_test):
    """
    Compare multiple classification models using LazyClassifier.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
    
    Returns:
        dict: Models comparison results
    """
    try:
        from lazypredict.Supervised import LazyClassifier
        
        print("🔍 Starting model comparison with LazyClassifier...")
        print(f"   Training set: {X_train.shape}")
        print(f"   Testing set: {X_test.shape}")
        
        # Initialize LazyClassifier
        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None
        )
        
        # Fit and predict
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        
        print(f"✅ Model comparison completed")
        print(f"   Total models tested: {len(models)}")
        
        return {
            'models': models,
            'predictions': predictions,
            'models_count': len(models)
        }
    except ImportError:
        print("⚠️ LazyClassifier not installed. Install with: pip install lazypredict")
        return None
    except Exception as e:
        print(f"❌ Error in model comparison: {e}")
        return None


def format_models_comparison(comparison_dict):
    """
    Format model comparison results into readable format.
    
    Args:
        comparison_dict: Results from compare_models_lazy_classifier
    
    Returns:
        dict: Formatted results with HTML table
    """
    try:
        if not comparison_dict or 'models' not in comparison_dict:
            return None
        
        models_df = comparison_dict['models']
        
        # Ensure dataframe is properly formatted
        if isinstance(models_df, pd.DataFrame):
            # Round numeric columns
            display_df = models_df.copy()
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].round(4)
            
            # Sort by Accuracy (descending)
            if 'Accuracy' in display_df.columns:
                display_df = display_df.sort_values('Accuracy', ascending=False)
            
            # Create HTML table
            html_table = display_df.to_html(
                classes='table table-striped table-hover table-sm',
                float_format=lambda x: f'{x:.4f}'
            )
            
            # Get top models
            top_5_models = display_df.head(5)
            
            print(f"✅ Models formatted for display")
            print(f"   Top model: {display_df.index[0]}")
            
            return {
                'all_models_html': html_table,
                'all_models_df': display_df,
                'top_5_models': top_5_models.to_dict('index'),
                'top_model_name': display_df.index[0],
                'top_model_accuracy': display_df['Accuracy'].iloc[0] if 'Accuracy' in display_df.columns else None,
                'models_count': len(display_df)
            }
        else:
            print("⚠️ Models format not recognized")
            return None
    except Exception as e:
        print(f"❌ Error formatting results: {e}")
        return None


def plot_models_comparison(comparison_dict, metric='Accuracy', top_n=15):
    """
    Create visualization comparing model performance.
    
    Args:
        comparison_dict: Results from format_models_comparison
        metric: Metric to use for comparison (default: Accuracy)
        top_n: Number of top models to display
    
    Returns:
        dict: Base64 encoded image
    """
    try:
        if not comparison_dict or 'all_models_df' not in comparison_dict:
            return None
        
        models_df = comparison_dict['all_models_df']
        
        if metric not in models_df.columns:
            print(f"⚠️ Metric '{metric}' not found. Available: {models_df.columns.tolist()}")
            return None
        
        # Get top N models
        top_models = models_df.nlargest(top_n, metric)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(top_models)), top_models[metric].values, color='steelblue')
        plt.yticks(range(len(top_models)), top_models.index)
        plt.xlabel(metric, fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Models by {metric}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Color bars based on performance
        colors = plt.cm.RdYlGn(top_models[metric].values / top_models[metric].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, v in enumerate(top_models[metric].values):
            plt.text(v - 0.01, i, f'{v:.4f}', va='center', ha='right', fontweight='bold', color='white')
        
        plt.xlim(0, max(top_models[metric].values) * 1.1)
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        print(f"✅ Comparison plot generated for metric: {metric}")
        
        return {
            'image_base64': f'data:image/png;base64,{img_base64}',
            'metric': metric,
            'top_n': top_n
        }
    except Exception as e:
        print(f"❌ Error plotting comparison: {e}")
        return None


def get_model_recommendations(comparison_dict):
    """
    Generate recommendations based on model comparison.
    
    Args:
        comparison_dict: Results from format_models_comparison
    
    Returns:
        dict: Recommendations and insights
    """
    try:
        if not comparison_dict or 'all_models_df' not in comparison_dict:
            return None
        
        models_df = comparison_dict['all_models_df']
        
        recommendations = {
            'best_model': comparison_dict.get('top_model_name'),
            'best_model_accuracy': comparison_dict.get('top_model_accuracy'),
            'total_models_tested': comparison_dict.get('models_count'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate statistics
        if 'Accuracy' in models_df.columns:
            accuracy_stats = {
                'max_accuracy': round(models_df['Accuracy'].max(), 4),
                'min_accuracy': round(models_df['Accuracy'].min(), 4),
                'mean_accuracy': round(models_df['Accuracy'].mean(), 4),
                'std_accuracy': round(models_df['Accuracy'].std(), 4)
            }
            recommendations['accuracy_stats'] = accuracy_stats
        
        # Get precision scores
        if 'Precision' in models_df.columns:
            precision_stats = {
                'max_precision': round(models_df['Precision'].max(), 4),
                'mean_precision': round(models_df['Precision'].mean(), 4)
            }
            recommendations['precision_stats'] = precision_stats
        
        # Get recall scores
        if 'Recall' in models_df.columns:
            recall_stats = {
                'max_recall': round(models_df['Recall'].max(), 4),
                'mean_recall': round(models_df['Recall'].mean(), 4)
            }
            recommendations['recall_stats'] = recall_stats
        
        # Identify fast vs accurate models
        if 'Time Taken' in models_df.columns and 'Accuracy' in models_df.columns:
            # Find fastest models (top 5)
            fastest = models_df.nsmallest(5, 'Time Taken')
            recommendations['fastest_models'] = fastest.index.tolist()
            
            # Find most accurate models (top 5)
            most_accurate = models_df.nlargest(5, 'Accuracy')
            recommendations['most_accurate_models'] = most_accurate.index.tolist()
        
        print(f"✅ Generated recommendations")
        print(f"   Best model: {recommendations['best_model']}")
        print(f"   Best accuracy: {recommendations['best_model_accuracy']:.4f}")
        
        return recommendations
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return None


def export_model_comparison_report(comparison_dict, output_path=None):
    """
    Export model comparison as CSV or JSON.
    
    Args:
        comparison_dict: Results from format_models_comparison
        output_path: Path to save the report (CSV format)
    
    Returns:
        bool: Success status
    """
    try:
        if not comparison_dict or 'all_models_df' not in comparison_dict:
            return False
        
        models_df = comparison_dict['all_models_df']
        
        if output_path:
            if output_path.endswith('.csv'):
                models_df.to_csv(output_path)
                print(f"✅ Model comparison exported to: {output_path}")
            elif output_path.endswith('.json'):
                models_df.to_json(output_path, orient='index')
                print(f"✅ Model comparison exported to: {output_path}")
            else:
                models_df.to_csv(output_path + '.csv')
                print(f"✅ Model comparison exported to: {output_path}.csv")
        
        return True
    except Exception as e:
        print(f"❌ Error exporting report: {e}")
        return False
