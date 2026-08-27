import sys
import xlrd  # Ensure this is installed if working with older .xls files
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectFromModel   
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    # ConfusionMatrixDisplay,
    f1_score
)
# # Append project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.prediction_service import predict_default,batch_predict_default

# filepath = "C:/Users/masra/Desktop/Project/credit-card-prediction/data/default of credit card clients.xls"
def preprocess_data(df):
    print("🔍 Preprocessing data...")
    output_model_dir="models"
    # df = pd.read_excel(filepath, header=1)#coment this later
    # Coerce to numeric, force invalid values to NaN (optional: handle NaNs later)
    col_meta =  [
        'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4'
    ]
    # Check if the dataset has proper column names
    if 'PAY_0' not in df.columns:
        print("Renaming generic columns to meaningful names...")

        # Rename columns
        df.columns = col_meta
    else:
        print("Column names already descriptive. No renaming needed.")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional: Handle NaNs if coercion failed on any row
    df.dropna(subset=col_meta, inplace=True)
    for col in col_meta:
        if col not in df.columns:
            print(f"Column {col} is missing from the dataset. Please check the input file.")
            sys.exit(1)

    if col not in df.columns:
        print(f"Missing required column: {col}")
        sys.exit(1)  # Stop script if any required column is missing

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    # Rule 1: Serious delay (>=2) in any payment status
    serious_delay = (
        (df['PAY_0'] >= 2) |
        (df['PAY_2'] >= 2) |
        (df['PAY_3'] >= 2) |
        (df['PAY_4'] >= 2) |
        (df['PAY_5'] >= 2) |
        (df['PAY_6'] >= 2)
    )

    # Rule 2: Low repayment ratio (average pay < 20% of bill)
    bill_total = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
    pay_total = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4']].mean(axis=1)
    low_payment_ratio = (pay_total / (bill_total + 1e-6)) < 0.2  # +1e-6 avoids division by zero

    # Rule 3: Low credit limit and delay
    low_limit = df['LIMIT_BAL'] < 100000
    some_delay = (df['PAY_0'] > 0)

    if 'default.payment.next.month' not in df.columns:
    # Initialize the target variable with zeros
        df['default.payment.next.month'] = 0
    # Apply the rules to generate the target variable
        df['default.payment.next.month'] = ((serious_delay) | (low_payment_ratio) | (low_limit & some_delay)).astype(int)
        print("Generated target variable:")

    df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
    print("✅ Preprocessing complete. Target variable 'default.payment.next.month' generated.")

    if df.isnull().values.any():
        print("✅ Checking for missing values...")
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
    
    # Check for the target column
    target_column = 'default'  # Change this if your target is named differently
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("✅ Features and target variable separated.")
    proc_file, run_dir = train_model(df, X, y, output_model_dir=output_model_dir)
    print("✅ Data preprocessing and model training complete.")
    return proc_file, run_dir

def train_model(df, X, y, output_model_dir):
    """
    Train a machine learning model.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        output_model_dir (str): Directory to save the trained model.
    """
    # Handling missing values (if any)
    imputer = SimpleImputer(strategy='median')  # or strategy='mean'
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Print final shapes
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    # -----------------------------
    # ✅ Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # ✅ Handling over sampling Class Imbalance with SMOTE    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)      
    print("✅ Class imbalance handled with SMOTE.")
    # -----------------------------
    # ✅ Feature Selection using Random Forest  
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_resampled, y_train_resampled)
    X_train_selected = selector.transform(X_train_resampled)
    X_test_selected = selector.transform(X_test_scaled)
    print("✅ Feature selection complete.")
    # -----------------------------
    # ✅ Model Training 
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train_selected, y_train_resampled)  
    # --- 3. Train Model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- 4. Predict Probabilities ---
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)

    # ROC Curve + AUC

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig_roc = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.savefig(f"static/roc_curve_{ts}.png")
    plt.show()


    # Precision-Recall Curve

    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    prc_fig = plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.show()
    # F1 Score
    f1 = f1_score(y_test, y_preds)
    print(f"F1 Score: {f1:.2f}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_preds)
    print("Confusion Matrix:")
    print(cm)
    # Optional: Classification Report
    print("Classification Report:")
    clas_report = classification_report(y_test, y_preds)
    print(clas_report)
    print("✅ Model training complete.")
          
    artifact_paths = save_artifacts(
        model=model,
        scaler=scaler,
        train_scaled= X_train_scaled,
        sample_input_df=X.iloc[:10].copy(),  # raw input for reference
        full_df=df,                           # full cleaned dataset
        clas_report=clas_report,
        roc_curve=fig_roc,
        precision_recall_curve=prc_fig,
        conf_matr = cm,
        prefix="credit_model",
    )
    print("✅ Artifacts saved run dir:", artifact_paths['run_dir'])
    print(f"✅ Model & scaler saved to", artifact_paths['model'], artifact_paths['scaler'])
    return artifact_paths["preprocessed_data"], artifact_paths['run_dir'] # Return the path to the full preprocessed DataFrame



def save_artifacts(model, scaler, train_scaled, sample_input_df, full_df=None, clas_report=None, roc_curve=None, precision_recall_curve=None, conf_matr = None, prefix="credit_model"):
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
    # run_dir = os.path.join("models", f"run_{timestamp}")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(BASE_DIR, "models", run_name)
    # run_dir = os.path.join(output_model_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{prefix}_{timestamp}.pkl")
    scaler_path = os.path.join(run_dir, f"scaler_{timestamp}.pkl")
    train_scaled_path = os.path.join(run_dir, f"X_train_scaled_{timestamp}.pkl")
    sample_input_path = os.path.join(run_dir, f"sample_batch_input.csv")
    df_path = None

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(train_scaled, train_scaled_path)

    # Save sample input (first few rows from raw features)
    sample_input_df.to_csv(sample_input_path, index=False)

    # Save full preprocessed DataFrame, if provided
    if full_df is not None:
        df_path = os.path.join(run_dir, f"preprocessed_data.csv")
        full_df.to_csv(df_path, index=False)
    # if clas_report is not None:
    #     clas_report_path = os.path.join(run_dir, f"classification_report_{timestamp}.txt")
    #     with open(clas_report_path, 'w') as f:
    #         f.write(clas_report)
    #     print(f"✅ Classification report saved to {clas_report_path}")  
    
    # ✅ Save ROC Curve image if figure provided
    # roc_curve = None
    if roc_curve is not None:
        roc_curve_path = os.path.join(run_dir, f"roc_curve.png")
        roc_curve.savefig(roc_curve_path)
        print(f"✅ ROC Curve saved at {roc_curve_path}")
    # ✅ Save Precision-Recall Curve image if figure provided
    if precision_recall_curve is not None:
        prc_curve_path = os.path.join(run_dir, f"precision_recall_curve.png")
        precision_recall_curve.savefig(prc_curve_path)
        print(f"✅ Precision-Recall Curve saved at {prc_curve_path}")
       # ✅ Save Confusion Matrix and Classification Report into one text file
    if conf_matr is not None or clas_report is not None:
        metrics_txt_path = os.path.join(run_dir, f"model_metrics.txt")
        with open(metrics_txt_path, "w") as f:
            if conf_matr is not None:
                f.write("Confusion Matrix:\n")
                f.write('----------------------------------------\n')
                f.write('********************************************\n')
                for row in conf_matr:
                    f.write(" ".join(map(str, row)) + "\n")
                f.write("\n")
                f.write('----------------------------------------\n')
                f.write('********************************************\n')
            if clas_report is not None:
                f.write("Classification Report:\n")
                f.write('----------------------------------------\n')
                f.write(clas_report)
                f.write('----------------------------------------\n')
        print(f"✅ Confusion Matrix and Classification Report saved at {metrics_txt_path}")


    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")

    print(f"✅ Artifacts saved in: {run_dir}")

    return {
        "run_dir": run_dir,
        "sample_input": sample_input_path,
        "model": model_path,
        "scaler": scaler_path,
        "train_scaled": train_scaled_path ,
        "preprocessed_data": df_path,
        "classification_report": clas_report
    }

# if __name__ == "__main__":
#     preprocess_data(filepath)
