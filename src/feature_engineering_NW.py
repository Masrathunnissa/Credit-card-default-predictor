import sys
import xlrd  # Ensure this is installed if working with older .xls files
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectFromModel   
from scipy import stats

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction_service import predict_default,batch_predict_default

filepath = "C:/Users/masra/Desktop/Project/credit-card-prediction/data/default of credit card clients.xls"
def preprocess_data(filepath):
    """
    Perform feature engineering on the given DataFrame.
    This function will:
    - Rename columns
    - Handle missing values
    - Convert categorical variables to numerical
    - Check for duplicates and remove them
    - Detect and handle outliers
    - Check for skewness and apply transformations if necessary
    - Check for multicollinearity and remove highly correlated features
    - Rename feature columns to 'feature_1', 'feature_2', etc.
    - Check for class imbalance and apply SMOTE if necessary
    - Ensure target variable is binary (0/1)
    """
       
    print("🔍 Preprocessing data...")
    output_model_dir="models"
    df = pd.read_excel(filepath, header=1)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    print("✅ Checking for missing values...")

    if df.isnull().values.any():
        df.fillna(df.mean(), inplace=True)
        print("⚠️ Missing values found and filled with column means.")

    # Convert categorical variables to numerical
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() < 10:
            df[col] = pd.Categorical(df[col]).codes
            print(f"✅ Converted categorical column '{col}' to numerical codes.")
        else:
            print(f"⚠️ Column '{col}' has too many unique values to convert to numerical codes.")
    
    print("✅ Checking for duplicates...")
        # Check for duplicates
    df.drop_duplicates(inplace=True)

    # # Check for outliers using Z-score
    # z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    # outliers = (z_scores > 3).any(axis=1)   
    # if outliers.any():
    #     print("⚠️ Outliers detected and removed.")
    #     df = df[~outliers]

    # # Check for skewness    
    # skewed_cols = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
    # for col in skewed_cols[skewed_cols > 1].index:
    #     df[col] = np.log1p(df[col])

    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(df.mean(), inplace=True)
    # above replacement 
    # for col in skewed_cols[skewed_cols > 1].index:
    # if (df[col] <= -1).any():
    #     print(f"⚠️ Skipping log1p on '{col}' due to values ≤ -1.")
    #     continue
    # df[col] = np.log1p(df[col])


    # # Check for multicollinearity
    # correlation_matrix = df.corr()
    # np.fill_diagonal(correlation_matrix.values, 0)
    # to_drop = [col for col in correlation_matrix.columns if any(correlation_matrix[col].abs() > 0.8)]
    # df.drop(columns=to_drop, inplace=True)
    # print(f"✅ Removed columns: {to_drop}")

    # if any(np.abs(correlation_matrix) > 0.8):
    #     print("⚠️ Multicollinearity detected. Removing highly correlated features.")
    #     to_drop = set()
    #     for i in range(len(correlation_matrix.columns)):
    #         for j in range(i):
    #             if abs(correlation_matrix.iloc[i, j]) > 0.8:
    #                 colname = correlation_matrix.columns[i]
    #                 to_drop.add(colname)
    #     df.drop(columns=to_drop, inplace=True)  
    #     print(f"✅ Removed columns: {to_drop}")

    # Rename feature columns to 'feature_1', 'feature_2', ...
    feature_cols = [col for col in df.columns if col != 'default']
    # new_names = {col: f"feature_{i+1}" for i, col in enumerate(feature_cols)}
    # df.rename(columns=new_names, inplace=True)
    new_feature_names = [f'feature_{i+1}' for i in range(len(feature_cols))]
    rename_dict = dict(zip(feature_cols, new_feature_names))
    df.rename(columns=rename_dict, inplace=True)
    # # Get column names of selected features
    # selected_feature_names = df.drop("default", axis=1).columns[selected_features]
    # df = pd.concat([df[selected_feature_names], df["default"]], axis=1)

        # Check for class imbalance
    if df['default'].value_counts(normalize=True).min() < 0.1:
        print("⚠️ Class imbalance detected. Applying SMOTE.")
        smote = SMOTE(random_state=42)
        X, y = df.drop("default", axis=1), df["default"]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df = pd.concat([X_resampled, y_resampled], axis=1)

    # Ensure target is integer and only contains 0/1
    df['default'] = df['default'].round().astype(int)

    # Check for feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df.drop("default", axis=1), df["default"])
    selector = SelectFromModel(model, prefit=True, threshold="mean")
    selected_features = selector.get_support(indices=True)
    # selected_cols = df.drop("default", axis=1).iloc[:, selected].columns.tolist()
    # df = df[selected_cols + ['default']]
    if len(selected_features) < df.shape[1] - 1:  # If not all features are selected
        print("⚠️ Some features are not important. Reducing feature set.")      
        df = df.iloc[:, selected_features.tolist() + [-1]]  # Keep only selected features and target
    else:   
        print("✅ All features are important. No reduction needed.")    
    # Save cleaned data
    #     # ==== UNIQUE FILE LOGIC ====
    # # Generate unique ID (timestamp)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # Create a new subdirectory inside 'models/' for this run
    # run_dir = os.path.join(output_model_dir, f"run_{timestamp}")
    # os.makedirs(run_dir, exist_ok=True)
    # df_path = os.path.join(run_dir, f"{timestamp}_cleaned_credit_data.csv")

  
    # # Save the DataFrame to CSV
    # df.to_csv(df_path, index=False)
    print(f"✅ Preprocessed data calling Training")
    filepath = train_model(df)
    # df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    # results = [
    #     {
    #         'Prediction': 'Default' if predict_default(row.tolist())[0] == 1 else 'No Default',
    #         'Probability (%)': round(predict_default(row.tolist())[1] * 100, 2)
    #     }
    #     for _, row in df.iterrows()
    # ]
    return filepath

def train_model(df):
    print("🚀 Training model...")
    output_model_dir="models"
    # Features and target
    X = df.drop("default", axis=1)
    y = df["default"]
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ==== UNIQUE FILE LOGIC ====
    # # Generate unique ID (timestamp)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # Create a new subdirectory inside 'models/' for this run
    # run_dir = os.path.join(output_model_dir, f"run_{timestamp}")
    # os.makedirs(run_dir, exist_ok=True)
    # # Save model and scaler
    # model_path = os.path.join(run_dir, f"credit_model_{timestamp}.pkl")
    # scaler_path = os.path.join(run_dir, f"scaler_{timestamp}.pkl")
    # joblib.dump(model, model_path)
    # joblib.dump(scaler, scaler_path)
   
    # Save sample batch input CSV (optional but useful for testing batch endpoint)
    # sample_input = pd.DataFrame(X[:10], columns=X.columns)
    # csv_path = os.path.join(output_dir, f"sample_batch_input_{timestamp}.csv")
    # sample_input.to_csv(csv_path, index=False)
    # pd.DataFrame(X.iloc[:10], columns=X.columns).to_csv(
    #     os.path.join(run_dir, f"sample_batch_input_{timestamp}.csv"), index=False
    # )
    artifact_paths = save_artifacts(
        model=model,
        scaler=scaler,
        sample_input_df=X.iloc[:10].copy(),  # raw input for reference
        full_df=df                           # full cleaned dataset
    )

    print(f"✅ Model & scaler saved to",artifact_paths['model'])
    return artifact_paths["run_dir"]  # Return the path to the full preprocessed DataFrame


def save_artifacts(model, scaler, sample_input_df, full_df=None, prefix="credit_model"):
    """
    Save model, scaler, sample input, and optionally the full preprocessed DataFrame to a timestamped run directory.

    Args:
        run_dir (str): The directory to save files in.
        model (sklearn model): The trained model.
        scaler (sklearn scaler): The fitted scaler.
        sample_input_df (pd.DataFrame): A small sample of input features.
        full_df (pd.DataFrame, optional): The full cleaned dataset (with target).
        prefix (str): Prefix for filenames.
    Returns:
        dict: Dictionary of saved file paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("models", f"run_{timestamp}")
    # run_dir = os.path.join(output_model_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{prefix}_{timestamp}.pkl")
    scaler_path = os.path.join(run_dir, f"scaler_{timestamp}.pkl")
    sample_input_path = os.path.join(run_dir, f"sample_batch_input_{timestamp}.csv")
    df_path = None

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save sample input (first few rows from raw features)
    sample_input_df.to_csv(sample_input_path, index=False)

    # Save full preprocessed DataFrame, if provided
    if full_df is not None:
        df_path = os.path.join(run_dir, f"preprocessed_data_{timestamp}.csv")
        full_df.to_csv(df_path, index=False)

    print(f"✅ Artifacts saved in: {run_dir}")

    return {
        "run_dir": run_dir,
        "model": model_path,
        "scaler": scaler_path,
        "sample_input": sample_input_path,
        "preprocessed_data": df_path
    }
# def main(filepath, output_model_dir="models"):
#     """
#     Main function to preprocess data, train model, and save artifacts.
#     """
#     df, run_dir = preprocess_data(filepath, output_model_dir)
#     run_dir = train_model(df, output_model_dir)
    
#     # Load the trained model and scaler for saving artifacts
#     model = joblib.load(os.path.join(run_dir, f"credit_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"))
#     scaler = joblib.load(os.path.join(run_dir, f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"))
    
#     # Save artifacts
#     sample_input_df = df.drop("default", axis=1).head(10)  # Sample input for batch endpoint
#     save_artifacts(run_dir, model, scaler, sample_input_df, full_df=df)     
#     print("✅ All steps completed successfully!")
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Preprocess credit data, train model, and save artifacts.")
#     parser.add_argument("filepath", type=str, help="Path to the input Excel file.")
#     parser.add_argument("--output_model_dir", type=str, default="models", help="Directory to save model artifacts.")
    
#     args = parser.parse_args()

#     main(args.filepath, args.output_model_dir)