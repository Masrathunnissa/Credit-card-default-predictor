import os
import joblib
import numpy as np
from pathlib import Path    
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# input_list = [('limit_balance', '4145'), ('Gender', '1'), ('education', '1'), ('age', '20'), ('PAY_0', '713'), ('PAY_2', '866'), ('PAY_1', '762'), ('PAY_3', '456'), ('PAY_4', '522'), ('PAY_5', '987'), ('PAY_6', '789'), ('bill_amt_sep', '865'), ('bill_amt_aug', '832'), ('bill_amt_jul', '214'), ('bill_amt_jun', '542'), ('bill_amt_may', '789'), ('bill_amt_apr', '766'), ('pay_amt_sep', '721'), ('pay_amt_aug', '420'), ('pay_amt_jul', '102'), ('pay_amt_jun', '200'), ('pay_amt_may', '300'), ('pay_amt_mar', '400'), ('output_format', 'csv'), ('email', '')]
expected_features = [
    'limit_balance', 'Gender', 'education', 'age',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun', 'bill_amt_may', 'bill_amt_apr',
    'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun', 'pay_amt_may'
]


def preprocess_single_row(input_list):
    """
    Preprocess a single user input row (from the web form).
    Assumes the input is in the correct order and cleaned as float values.
    Returns a scaled numpy array ready for prediction.
    """
    # Load latest scaler
    latest_scaler_path = sorted(
        [os.path.join("models", d, f) for d in os.listdir("models") if d.startswith("run_") 
         for f in os.listdir(os.path.join("models", d)) if f.startswith("scaler")],
        reverse=True
    )[0]
    
    scaler = joblib.load(latest_scaler_path)
    
    # Convert to DataFrame with correct shape
    input_array = np.array(input_list).reshape(1, -1)
    
    # Scale it
    scaled_input = scaler.transform(input_array)
    print(f"✅ Scaled input: {scaled_input}")
    return scaled_input
