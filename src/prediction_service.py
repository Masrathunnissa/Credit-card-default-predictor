import os
import uuid
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from shap import force_plot
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fpdf import FPDF
from datetime import datetime

# input_array = [('limit_balance', '124552'), ('Gender', '1'), ('education', '1'), ('age', '30'), ('PAY_0', '1'), ('PAY_2', '2'), ('PAY_1', '4'), ('PAY_3', '5'), ('PAY_4', '6'), ('PAY_5', '4'), ('PAY_6', '5'), ('bill_amt_sep', '15482'), ('bill_amt_aug', '454541'), ('bill_amt_jul', '45545'), ('bill_amt_jun', '15454'), ('bill_amt_may', '4545'), ('bill_amt_apr', '4554'), ('pay_amt_sep', '1'), ('pay_amt_aug', '2'), ('pay_amt_jul', '2'), ('pay_amt_jun', '3'), ('pay_amt_may', ''), ('pay_amt_mar', ''), ('output_format', 'csv'), ('email', '')]
# === Helper: Find latest model directory ===
fields = [
            'limit_balance', 'Gender', 'education', 'age', 
            #'marital_status', 'Credit Card Type', 'Credit Card Limit Utilization', 'Credit Card Usage',
            'PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun', 'bill_amt_may', 'bill_amt_apr',
            'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun', 'pay_amt_may', 'pay_amt_mar'
        ]
optional_fields = ['Gender', 'education','age']


def format_user_input_lines(user_values=None, feature_names=None):
    if not user_values:
        return []

    selected_features = list(feature_names or fields)
    lines = []

    for feature_name in selected_features:
        if feature_name not in user_values:
            continue

        raw_value = user_values[feature_name]
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else ""

        value = str(raw_value).strip()
        if value:
            lines.append(f"{feature_name}: {value}")

    return lines


def parse_metrics_file(metrics_file_path):
    """Parse model_metrics.txt file and extract confusion matrix and classification report."""
    metrics_data = {
        'confusion_matrix': None,
        'classification_report': None,
        'accuracy': None,
        'tn': None, 'fp': None, 'fn': None, 'tp': None
    }
    
    try:
        with open(metrics_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract confusion matrix
        cm_start = content.find('Confusion Matrix:')
        cm_end = content.find('Classification Report:')
        if cm_start != -1 and cm_end != -1:
            cm_section = content[cm_start:cm_end]
            lines = [l.strip() for l in cm_section.split('\n') if l.strip() and not l.startswith('-') and not l.startswith('*')]
            matrix_lines = [l for l in lines if l and l[0].isdigit()]
            if len(matrix_lines) >= 2:
                tn, fp = map(int, matrix_lines[0].split())
                fn, tp = map(int, matrix_lines[1].split())
                metrics_data['confusion_matrix'] = [[tn, fp], [fn, tp]]
                metrics_data['tn'] = tn
                metrics_data['fp'] = fp
                metrics_data['fn'] = fn
                metrics_data['tp'] = tp
        
        # Extract classification report
        cr_start = content.find('Classification Report:')
        if cr_start != -1:
            cr_section = content[cr_start:]
            metrics_data['classification_report'] = cr_section
            # Extract accuracy
            if 'accuracy' in cr_section:
                acc_line = [l for l in cr_section.split('\n') if 'accuracy' in l]
                if acc_line:
                    acc_val = acc_line[0].split()[-2]
                    metrics_data['accuracy'] = float(acc_val)
    
    except Exception as e:
        print(f"Error parsing metrics file: {e}")
    
    return metrics_data


def generate_html_metrics_report(metrics_file_path, output_html_path):
    """Generate a beautiful HTML report from metrics file."""
    metrics = parse_metrics_file(metrics_file_path)
    tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
    accuracy = metrics['accuracy']
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Metrics Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 15px;
                padding: 40px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 1200px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 20px;
            }}
            .header h1 {{
                color: #333;
                font-weight: bold;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .metric-card .value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-card .label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .confusion-section, .report-section {{
                background: #f8f9fa;
                border-left: 5px solid #667eea;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .confusion-section h3, .report-section h3 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .confusion-matrix {{
                display: grid;
                grid-template-columns: auto auto auto;
                gap: 20px;
                max-width: 400px;
                margin-bottom: 20px;
            }}
            .cm-cell {{
                padding: 20px;
                text-align: center;
                border-radius: 8px;
                font-weight: bold;
            }}
            .cm-label {{
                background: #e9ecef;
                color: #333;
            }}
            .cm-tn {{
                background: #d4edda;
                color: #155724;
                font-size: 1.5em;
            }}
            .cm-fp {{
                background: #fff3cd;
                color: #856404;
                font-size: 1.5em;
            }}
            .cm-fn {{
                background: #fff3cd;
                color: #856404;
                font-size: 1.5em;
            }}
            .cm-tp {{
                background: #d4edda;
                color: #155724;
                font-size: 1.5em;
            }}
            .report-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .report-table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            .report-table td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            .report-table tr:nth-child(even) {{
                background: #f9fafb;
            }}
            .report-table tr:hover {{
                background: #f0f1ff;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                color: #999;
                font-size: 0.9em;
            }}
            .accuracy-bar {{
                background: #e9ecef;
                border-radius: 5px;
                overflow: hidden;
                margin: 10px 0;
                height: 30px;
            }}
            .accuracy-fill {{
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Model Performance Metrics Report</h1>
                <p class="text-muted">Comprehensive evaluation of model predictions</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Accuracy</div>
                    <div class="value">{accuracy*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sensitivity (Recall)</div>
                    <div class="value">{sensitivity*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Specificity</div>
                    <div class="value">{specificity*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Precision</div>
                    <div class="value">{precision*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">F1 Score</div>
                    <div class="value">{f1:.2f}</div>
                </div>
            </div>
            
            <div class="confusion-section">
                <h3>🔍 Confusion Matrix</h3>
                <p class="text-muted">Shows True Positives, False Positives, False Negatives, True Negatives</p>
                <div class="confusion-matrix">
                    <div class="cm-cell cm-label">Predicted →</div>
                    <div class="cm-cell cm-label">No Default</div>
                    <div class="cm-cell cm-label">Default</div>
                    <div class="cm-cell cm-label">Actual ↓</div>
                    <div class="cm-cell cm-tn">TN: {tn}</div>
                    <div class="cm-cell cm-fp">FP: {fp}</div>
                    <div class="cm-cell cm-label">No Default</div>
                    <div class="cm-cell cm-fn">FN: {fn}</div>
                    <div class="cm-cell cm-tp">TP: {tp}</div>
                    <div class="cm-cell cm-label">Default</div>
                </div>
                <p class="text-muted mt-3"><strong>Legend:</strong> TN = True Negatives, FP = False Positives, FN = False Negatives, TP = True Positives</p>
            </div>
            
            <div class="accuracy-section" style="margin-bottom: 30px;">
                <h4>Overall Accuracy Score</h4>
                <div class="accuracy-bar">
                    <div class="accuracy-fill" style="width: {accuracy*100}%;">
                        {accuracy*100:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="report-section">
                <h3>📋 Classification Report Details</h3>
                <pre style="background: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
{metrics['classification_report']}
                </pre>
            </div>
            
            <div class="footer">
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Credit Card Default Prediction Model | © 2026</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML metrics report generated: {output_html_path}")
    return output_html_path


def get_latest_model_dir(models_root="models"):
    # subdirs = [
    #     os.path.join(models_root, d) for d in os.listdir(models_root)
    #     if os.path.isdir(os.path.join(models_root, d)) and d.startswith("run_")
    # ]
    models_root = Path(models_root)
    subdirs = [d for d in models_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError("❌ No model runs found in 'models/' directory.")

    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

    # === Load latest model and scaler ===
latest_dir = get_latest_model_dir()
model_files = os.listdir(latest_dir)

model_path = [os.path.join(latest_dir, f) for f in model_files if "credit_model" in f][0]
scaler_path = [os.path.join(latest_dir, f) for f in model_files if "scaler" in f][0]
train_path = [os.path.join(latest_dir, f) for f in model_files if "X_train_scaled" in f][0]
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
X_train_scaled = joblib.load(train_path)

print("SCALER:", scaler)
print(f"✅ Loaded model from: {model_path}")
print(f"✅ Loaded scaler from: {scaler_path}")

# === Prediction function ===
# def com_batch_predict_default(input_data):
#     """
#     input_data: List or array of 23 numerical features
#     returns: (prediction, probability)
#     """
#     if len(input_data) != 23:
#         raise ValueError(f"Expected 23 features, got {len(input_data)}.")
#     input_array = np.array(input_data).reshape(1, -1)
#     input_scaled = scaler.transform(input_array)
#     prediction = model.predict(input_scaled)[0]
#     probability = model.predict_proba(input_scaled)[0][1] * 100  # Convert to percent
#     print(f"🔍 Prediction: {prediction}, Probability: {probability:.2f}%")
#     return prediction, round(probability, 2)

# def predict_from_dataframe_safe(df):
#     results = []
#     for i, (_, row) in enumerate(df.iterrows()):
#         try:
#             # cleaned_row = prepare_input_data(row)
#             # print(df.info())

#                 # ✅ Ensure all required columns are present
#             expected_cols = [col for col in fields if col != 'ID']  # Remove ID if it's already dropped
#             missing_cols = [col for col in expected_cols if col not in df.columns]

#             if missing_cols:
#                 print(f"⚠️ Missing columns found: {missing_cols}. Adding them with default 0s.")
#                 for col in missing_cols:
#                     df[col] = 0.0  # Default value as float64
#                 df = df.astype({col: 'float64' for col in missing_cols})

#             else:
#                 print("✅ All expected columns are present after preprocessing.")
#             # print(df.info())
#             prediction, prob  = batch_predict_default(df)
#             results.append({
#                 "Prediction": "Default" if prediction == 1 else "No Default",
#                 "Probability": prob
#             })
#         except Exception as e:
#             results.append({"error": f"Row {i} failed: {e}"})
#     return results

def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return os.path.basename(filename)

def save_results_to_pdf(results, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 10)
    y = height - 40  # Start from top

    c.drawString(30, y, "Credit Default Prediction Results")
    y -= 30

    for i, row in enumerate(results):
        line = f"{i+1}. " + ", ".join([f"{k}: {v}" for k, v in row.items()])
        c.drawString(30, y, line[:110])  # Truncate long lines
        y -= 15
        if y < 40:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 40

    c.save()
    print(f"Results saved to {filename}")
    return os.path.basename(filename)

def append_predictions_to_dataframe(input_df, results):
    predictions_df = pd.DataFrame(results)
    return input_df.reset_index(drop=True).join(predictions_df)

def predict_from_dataframe_safe(df):
    # Make sure all required columns exist
    expected_cols = [col for col in fields if col != 'ID']
    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}. Filling with 0s.")
        for col in missing_cols:
            df[col] = 0.0

    # Ensure numeric types
    df = df.astype({col: 'float64' for col in expected_cols if col in df.columns})

    print("✅ Columns and types ready. Running batch prediction...")
    return batch_predict_default(df)
def batch_predict_default(df):
    # Prepare input data (clean, reorder, etc.)
    input_array = prepare_input_data(df)

    # Scale all at once
    input_scaled = scaler.transform(input_array)

    # Predict all at once
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[:, 1] * 100

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        result = {
            "Row": i + 1,
            "Prediction": "Default" if pred == 1 else "No Default",
            "Probability": f"{prob:.2f}%"
        }

        # Include ID if available
        if 'ID' in df.columns:
            result['ID'] = df.iloc[i]['ID']

        results.append(result)
    
    return results
def prepare_batch_input_data(df):
    # Select and order columns
    expected_cols = [col for col in fields if col != 'ID']
    df = df.copy()

    # Only keep necessary columns
    df = df[expected_cols]

    # Fill NaNs if needed
    df = df.fillna(0)

    return df


# def batch_predict_default(input_dic):
#     input_array = prepare_input_data(input_dic)
#     if isinstance(input_array, pd.DataFrame):
#         input_array = input_array.values
#     elif isinstance(input_array, dict):
#         input_array = pd.DataFrame([input_array]).values
#     else:
#         raise ValueError("Input must be a DataFrame or dict")

#     # input_array = np.array(input_array).reshape(1, -1)
#     input_scaled = scaler.transform(input_array)

#     prediction = model.predict(input_scaled)[0]
#     probability = model.predict_proba(input_scaled)[0][1] * 100
#     # Save prediction as JSON or CSV
#     result = {
#         "prediction": int(prediction),
#         "probability": round(probability, 2)
#     }
#     return result

def com_predict_default(scaled_input):
    """
#     Predict default based on scaled input features.
#     """
    # Load latest model
    latest_model_path = sorted(
        [os.path.join("models", d, f) for d in os.listdir("models") if d.startswith("run_") 
         for f in os.listdir(os.path.join("models", d)) if f.startswith("credit_model")],
        reverse=True
    )[0]
    
    model = joblib.load(latest_model_path)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    print(f"🔍 Prediction: {prediction}, Probability: {probability:.2f}%")

    return prediction, probability

def predict_default(input_dic, user_values=None):
    input_df = prepare_input_data(input_dic)
    if isinstance(input_df, pd.DataFrame):
        input_array = input_df.to_numpy()
    elif isinstance(input_df, dict):
        input_array = pd.DataFrame([input_df]).to_numpy()
    else:
        raise ValueError("Input must be a DataFrame or dict")

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    result = {
        "prediction": int(prediction),
        "probability": round(probability, 2)
    }
    unique_id = uuid.uuid4().hex
    shap_filename = f"shap_{unique_id}.png"
    pdf_filename = f"single_report_{unique_id}.pdf"
    txt_filename = f"single_report_{unique_id}.txt"
    # Use app/static for all static files (where Flask serves from)
    static_dir = os.path.abspath(os.path.join("app", "static"))
    shap_plot_path = os.path.join(static_dir, shap_filename)
    pdf_path = os.path.join(static_dir, pdf_filename)
    txt_path = os.path.join(static_dir, txt_filename)
    os.makedirs(static_dir, exist_ok=True)

    save_shap_plot_as_png(
        model=model,
        input_scaled=input_scaled,
        shap_plot_path=shap_plot_path,
        X_train_scaled=input_scaled,
        feature_names=fields
    )

    print(f"SHAP plot saved to {shap_plot_path}")
    
    save_single_pdf(
        pdf_path=pdf_path,
        prediction=prediction,
        probability=probability,
        shap_plt=shap_plot_path,
        user_input_values=user_values,
        feature_names=fields
    )
    print(f"PDF report saved to {pdf_path}")
    
    save_single_txt(
        txt_path=txt_path,
        prediction=prediction,
        probability=probability,
        user_input_values=user_values,
        feature_names=fields
    )
    print(f"TXT report saved to {txt_path}")

    return prediction, probability, shap_filename, pdf_filename, txt_filename

from werkzeug.datastructures import ImmutableMultiDict
def clean_input(input_data):
    # Convert to dict if it's not already
    if not isinstance(input_data, ImmutableMultiDict):
        try:
            # input_data = dict(input_data)
            input_data = input_data.to_dict()
        except Exception as e:
            raise ValueError(f"Invalid input format: {e}")

    # Drop keys that are not features
    ignore_keys = ['output_format', 'email']
    input_data = {k: v for k, v in input_data.items() if k not in ignore_keys}
    # Step 1: Filter out fields not in `fields`
    filtered_input = {k: v for k, v in input_data.items() if k in fields}

    # Step 2: If still more than 21, drop optional fields in order
    while len(filtered_input) > 21 and optional_fields:
        for field in optional_fields:
            if field in filtered_input:
                del filtered_input[field]
                if len(filtered_input) == 21:
                    break

    # Step 3: Error if we still don’t have 21
    if len(filtered_input) != 21:
        raise ValueError(f"Expected 21 features, got {len(filtered_input)}. Final keys: {list(filtered_input.keys())}")
   
    # Step 4: Handle missing or blank values and convert to float
    for k in filtered_input:
        v = filtered_input[k]
        try:
            filtered_input[k] = float(v) if v != '' else 0.0
        except ValueError:
            raise ValueError(f"Invalid value for feature '{k}': {v}")
                          
    return filtered_input

def prepare_input_data(input_data):
    # Define required and optional fields
    # --- Handle row (Series or dict) ---
    if isinstance(input_data, (pd.Series, dict)):
        cleaned_row = clean_input(input_data)
        return pd.DataFrame([cleaned_row])

    elif isinstance(input_data, ImmutableMultiDict):
        cleaned_row = clean_input(input_data)
        return pd.DataFrame([cleaned_row])

    # --- Handle batch DataFrame ---
    elif isinstance(input_data, pd.DataFrame):
        cleaned_rows = []
        for idx, row in input_data.iterrows():
            try:
                cleaned_row = clean_input(row)
                cleaned_rows.append(cleaned_row)
            except Exception as e:
                raise ValueError(f"Row {idx} failed: {e}")
        return pd.DataFrame(cleaned_rows)

    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or dict.")

   
#    # Convert to DataFrame and cast to float
#     try:
#         input_df = pd.DataFrame([filtered_input])
#         input_df = input_df.astype(float)
#     except ValueError as ve:
#         raise ValueError(f"Invalid input. Ensure all fields are numeric. Error: {ve}")

#     return input_df

def save_shap_force_plot(model, input_scaled, X_train_scaled, shap_plot_path):
    #     # # SHAP explanation
    # explainer = shap.Explainer(model, input_scaled)  # Pass training data for background
    # shap_values = explainer(input_scaled)
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(input_scaled)

    # Save interactive HTML force plot using legacy API
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,  # single float for binary
        shap_values=shap_values.values[0],
        features=shap_values.data[0],
        feature_names=shap_values.feature_names
    )
    plt.show()
    # Save as HTML
    shap.save_html(shap_plot_path, force_plot)
    print(f"SHAP plot saved to html: {shap_plot_path}")

from fpdf import FPDF
from datetime import datetime
import os

def save_single_pdf(pdf_path, prediction=None, probability=None, shap_plt=None, user_input_values=None, feature_names=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Credit Default Prediction Report", ln=True, align='C')

    pdf.ln(5)

    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 8, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 8, txt="_"*90, ln=True, align='C')

    pdf.ln(5)

    # Prediction Result
    if prediction is not None:
        result_text = "HIGH RISK (Default)" if prediction == 1 else "LOW RISK (No Default)"
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(200, 8, txt=f"Prediction Result: {result_text}", ln=True)

    if probability is not None:
        pdf.set_font("Arial", '', 11)
        prob_display = f"{probability:.2f}%"
        pdf.cell(200, 7, txt=f"Probability of Default: {prob_display}", ln=True)

    pdf.cell(200, 8, txt="_"*90, ln=True, align='C')
    pdf.ln(5)

    # User Input Values Section
    if user_input_values:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 7, txt="Your Provided Information:", ln=True)
        pdf.set_font("Arial", '', 9)
        lines = format_user_input_lines(user_input_values, feature_names or fields)
        for line in lines:
            pdf.multi_cell(190, 5, txt=f"  - {line}")
        pdf.ln(3)

    pdf.cell(200, 7, txt="_"*90, ln=True, align='C')
    pdf.ln(5)

    # Explanation Section
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 7, txt="How This Result Was Calculated:\\n", ln=True)
    pdf.set_font("Arial", '', 9)
    explanation_text = (
        f"Based on the information you provided, our machine learning model analyzed {len(fields)} key features "
        f"related to your credit card usage patterns. The model was trained on historical credit card data to identify "
        f"patterns associated with payment defaults.\\n\\n"
        f"Key factors in this prediction include:\\n"
        f"  - Repayment history and payment status patterns (PAY_0 through PAY_6)\\n"
        f"  - Your credit limit and current balance amounts\\n"
        f"  - Monthly bill amounts and payment behavior\\n\\n"
        f"The SHAP plot below shows which features had the most influence on this prediction. "
        f"Longer bars indicate stronger feature impact on the prediction."
    )
    pdf.multi_cell(190, 4, txt=explanation_text)
    pdf.ln(5)

    pdf.cell(200, 7, txt="_"*90, ln=True, align='C')
    pdf.ln(5)

    # SHAP Plot
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 7, txt="Feature Contribution Analysis (SHAP Waterfall Plot):", ln=True)
    pdf.set_font("Arial", '', 9)
    shap_explanation = (
        "The SHAP waterfall plot below shows how each of your provided feature values affects the final prediction. "
        "It starts with a base value (average prediction) and shows cumulative contributions from each feature:\\n\\n"
        "HOW TO READ THE PLOT:\\n"
        "- RED BARS (pushing right): Features that INCREASE your default probability (higher risk)\\n"
        "- BLUE BARS (pushing left): Features that DECREASE your default probability (lower risk)\\n"
        "- BAR LENGTH: Longer bars mean stronger influence on the prediction\\n"
        "- FINAL VALUE: All bars combine to show your final prediction probability\\n\\n"
        "This explains WHICH features influenced your specific prediction and HOW they affected it.\n"
    )
    pdf.multi_cell(190, 4, txt=shap_explanation)
    pdf.ln(3)
    
    if shap_plt and os.path.exists(shap_plt):
        try:
            pdf.image(shap_plt, x=10, y=None, w=190)
        except Exception as e:
            pdf.set_font("Arial", '', 9)
            pdf.cell(200, 7, txt=f"(SHAP plot unavailable: {e})", ln=True)
    else:
        pdf.set_font("Arial", '', 9)
        pdf.cell(200, 7, txt="(SHAP plot not available)", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(200, 6, txt="Disclaimer: This prediction is based on historical patterns and should be used as guidance.", ln=True)

    pdf.ln(3)
    pdf.cell(200, 6, txt="@2026 Credit Card Default Prediction System | By Masrath Unnissa", ln=True, align='C')
    pdf.output(pdf_path)
    print(f"✅ PDF saved at: {pdf_path}")

def save_single_txt(txt_path, prediction=None, probability=None, user_input_values=None, feature_names=None):
    """Save prediction report as a plain text file."""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("CREDIT DEFAULT PREDICTION REPORT\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Prediction Result
        if prediction is not None:
            result_text = "HIGH RISK (Default)" if prediction == 1 else "LOW RISK (No Default)"
            f.write(f"PREDICTION RESULT: {result_text}\n")
        
        if probability is not None:
            f.write(f"Probability of Default: {probability:.2f}%\n")
        
        f.write("\n" + "=" * 90 + "\n\n")
        
        # User Input Values Section
        if user_input_values:
            f.write("YOUR PROVIDED INFORMATION:\n")
            f.write("-" * 90 + "\n")
            lines = format_user_input_lines(user_input_values, feature_names or fields)
            for line in lines:
                f.write(f"  - {line}\n")
            f.write("\n")
        
        f.write("=" * 90 + "\n\n")
        
        # Explanation Section
        f.write("HOW THIS RESULT WAS CALCULATED:\n")
        f.write("-" * 90 + "\n")
        explanation_text = (
            f"Based on the information you provided, our machine learning model analyzed {len(fields)} key features "
            f"related to your credit card usage patterns. The model was trained on historical credit card data to identify "
            f"patterns associated with payment defaults.\n\n"
            f"Key factors in this prediction include:\n"
            f"  - Repayment history and payment status patterns (PAY_0 through PAY_6)\n"
            f"  - Your credit limit and current balance amounts\n"
            f"  - Monthly bill amounts and payment behavior\n\n"
            f"The SHAP analysis shown in the PDF shows which features had the most influence on this prediction. "
            f"Longer bars indicate stronger feature impact on the prediction.\n"
            f"Red features increase default risk, while blue features decrease default risk.\n"
        )
        f.write(explanation_text)
        f.write("\n" + "=" * 90 + "\n\n")
        
        # SHAP Explanation
        f.write("FEATURE CONTRIBUTION ANALYSIS (SHAP Waterfall Plot Explanation):\n")
        f.write("-" * 90 + "\n")
        shap_explanation = (
            "The SHAP waterfall plot shows how each feature contributes to the final prediction:\n\n"
            "1. BASE VALUE: This is the average prediction across all historical data.\n"
            "2. FEATURE CONTRIBUTIONS: Each bar shows how much that feature pushes the prediction up or down.\n"
            "3. COLOR CODING:\n"
            "   - RED/NEGATIVE: Features that INCREASE the probability of default (higher risk)\n"
            "   - BLUE/POSITIVE: Features that DECREASE the probability of default (lower risk)\n"
            "4. MAGNITUDE: The length of each bar indicates how strong the influence is.\n"
            "5. CUMULATIVE EFFECT: The bars stack up to reach the final prediction probability.\n\n"
            "This analysis helps explain WHICH specific features influenced your prediction most strongly "
            "and in which direction (risk-increasing or risk-decreasing).\n"
        )
        f.write(shap_explanation)
        f.write("\n" + "=" * 90 + "\n\n")
        
        f.write("DISCLAIMER:\n")
        f.write("-" * 90 + "\n")
        f.write("This prediction is based on historical patterns and should be used as guidance only.\n")
        f.write("It does not guarantee future outcomes and should not be the sole basis for financial decisions.\n\n")
        
        f.write("@2026 Credit Card Default Prediction System | By Masrath Unnissa\n")
        f.write("=" * 90 + "\n")
    
    print(f"✅ TXT report saved at: {txt_path}")

def save_shap_plot_as_png(model, input_scaled, shap_plot_path, X_train_scaled=None, feature_names=None):
    """
    Save SHAP explanation plot for a single prediction as a PNG image using waterfall plot.
    """
    if X_train_scaled is not None:
        explainer = shap.Explainer(model, X_train_scaled)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(input_scaled)

    plt.figure(figsize=(14, 9))
    try:
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"SHAP Waterfall Plot - How Features Affect Default Prediction\n(Red = Increases Risk, Blue = Decreases Risk)", fontsize=13, fontweight='bold')
    except Exception as e:
        print(f"Waterfall plot failed: {e}")
        shap.plots.force(shap_values.base_values, shap_values.values[0], features=input_scaled[0], feature_names=feature_names or None, matplotlib=True, show=False)

    plt.tight_layout()
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"✅ SHAP plot saved at: {shap_plot_path}")


def save_batch_pdf(pdf_path, results, original_rows=0, processed_rows=0, output_format="csv"):
    """
    Save batch prediction results as a PDF report.
    
    Args:
        pdf_path: Path to save the PDF file
        results: List of prediction result dictionaries
        original_rows: Number of original rows in uploaded file
        processed_rows: Number of rows after preprocessing
        output_format: Format of output (csv, pdf, xlsx, etc.)
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=12)
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Credit Card Default Batch Prediction Report", ln=True, align='C')
        
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 8, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(200, 8, txt="_"*90, ln=True, align='C')
        
        pdf.ln(5)
        
        # Summary Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt="Batch Processing Summary", ln=True)
        
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 7, txt=f"  - Original Records: {original_rows}", ln=True)
        pdf.cell(200, 7, txt=f"  - Processed Records: {processed_rows}", ln=True)
        pdf.cell(200, 7, txt=f"  - Output Format: {output_format.upper()}", ln=True)
        pdf.cell(200, 7, txt=f"  - Total Predictions: {len(results)}", ln=True)
        
        # Count defaults
        defaults_count = sum(1 for r in results if r.get('Prediction') == 'Default')
        no_defaults_count = len(results) - defaults_count
        pdf.cell(200, 7, txt=f"  - Records with Default Risk: {defaults_count}", ln=True)
        pdf.cell(200, 7, txt=f"  - Records with No Default Risk: {no_defaults_count}", ln=True)
        
        pdf.ln(5)
        pdf.cell(200, 8, txt="_"*90, ln=True, align='C')
        pdf.ln(5)
        
        # Detailed Results Section (First 50 rows)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 8, txt="Detailed Prediction Results (First 50 records)", ln=True)
        
        pdf.set_font("Arial", '', 9)
        pdf.ln(3)
        
        # Display first 50 results
        for i, result in enumerate(results[:50]):
            line = f"{result.get('Row', i+1)}. Pred: {result.get('Prediction', 'N/A')} | Prob: {result.get('Probability', 'N/A')}"
            if 'ID' in result:
                line += f" | ID: {result['ID']}"
            pdf.cell(200, 5, txt=line[:95], ln=True)
        
        if len(results) > 50:
            pdf.set_font("Arial", 'I', 9)
            pdf.cell(200, 6, txt=f"... and {len(results) - 50} more records (see attached CSV/PDF file)", ln=True)
        
        pdf.ln(5)
        pdf.cell(200, 8, txt="_"*90, ln=True, align='C')
        pdf.ln(5)
        
        # Footer Info
        pdf.set_font("Arial", '', 9)
        pdf.multi_cell(190, 5, txt="For complete results, please refer to the attached predictions file in your selected output format (CSV/PDF).\n\n"
            "Disclaimer: These predictions are based on historical patterns and should be used as guidance only. "
            "They do not guarantee future outcomes.")
        
        pdf.ln(3)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(200, 6, txt="@2026 Credit Card Default Prediction System | By Masrath Unnissa", ln=True, align='C')
        
        pdf.output(pdf_path)
        print(f"✅ Batch PDF report saved at: {pdf_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving batch PDF: {e}")
        return False


def save_batch_txt(txt_path, results, original_rows=0, processed_rows=0, output_format="csv"):
    """
    Save batch prediction results as a TXT report.
    
    Args:
        txt_path: Path to save the TXT file
        results: List of prediction result dictionaries
        original_rows: Number of original rows in uploaded file
        processed_rows: Number of rows after preprocessing
        output_format: Format of output (csv, pdf, xlsx, etc.)
    """
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("CREDIT CARD DEFAULT BATCH PREDICTION REPORT\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary Section
            f.write("BATCH PROCESSING SUMMARY:\n")
            f.write("-" * 90 + "\n")
            f.write(f"  - Original Records: {original_rows}\n")
            f.write(f"  - Processed Records: {processed_rows}\n")
            f.write(f"  - Output Format: {output_format.upper()}\n")
            f.write(f"  - Total Predictions: {len(results)}\n")
            
            # Count defaults
            defaults_count = sum(1 for r in results if r.get('Prediction') == 'Default')
            no_defaults_count = len(results) - defaults_count
            f.write(f"  - Records with Default Risk: {defaults_count}\n")
            f.write(f"  - Records with No Default Risk: {no_defaults_count}\n")
            f.write(f"  - Default Rate: {(defaults_count/len(results)*100):.2f}%\n")
            f.write("\n" + "=" * 90 + "\n\n")
            
            # Detailed Results Section (First 50 rows)
            f.write("DETAILED PREDICTION RESULTS (First 50 records):\n")
            f.write("-" * 90 + "\n")
            for i, result in enumerate(results[:50]):
                f.write(f"Row {result.get('Row', i+1)}: ")
                f.write(f"Prediction={result.get('Prediction', 'N/A')}, ")
                f.write(f"Probability={result.get('Probability', 'N/A')}")
                if 'ID' in result:
                    f.write(f", ID={result['ID']}")
                f.write("\n")
            
            if len(results) > 50:
                f.write(f"\n... and {len(results) - 50} more records\n")
            
            f.write("\n" + "=" * 90 + "\n\n")
            
            # Footer Info
            f.write("NOTES:\n")
            f.write("-" * 90 + "\n")
            f.write("For complete results, please refer to the attached predictions file in your selected output format.\n\n")
            f.write("Prediction Categories:\n")
            f.write("  - 'Default': Model predicts payment default risk\n")
            f.write("  - 'No Default': Model predicts no default risk\n")
            f.write("  - Probability (%): Confidence level of the prediction\n\n")
            f.write("Disclaimer: These predictions are based on historical patterns and should be used as guidance only.\n")
            f.write("They do not guarantee future outcomes and should not be the sole basis for decisions.\n\n")
            f.write("@2026 Credit Card Default Prediction System | By Masrath Unnissa\n")
            f.write("=" * 90 + "\n")
        
        print(f"✅ Batch TXT report saved at: {txt_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving batch TXT: {e}")
        return False

# input_array = "C:/Users/masra/Desktop/Project/credit-card-prediction/models/run_20250715_232259/preprocessed_data.csv"
# if __name__ == "__main__":
#     df=pd.read_csv(input_array)
#     print(df.info())
#     col_meta =  [
#         'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
#         'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
#         'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
#         'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6'
#     ]
#         # ✅ Ensure all required columns are present
#     expected_cols = [col for col in col_meta if col != 'ID']  # Remove ID if it's already dropped
#     missing_cols = [col for col in expected_cols if col not in df.columns]

#     if missing_cols:
#         print(f"⚠️ Missing columns found: {missing_cols}. Adding them with default 0s.")
#         for col in missing_cols:
#             df[col] = 0.0  # Default value as float64
#         df = df.astype({col: 'float64' for col in missing_cols})

#     else:
#         print("✅ All expected columns are present after preprocessing.")
#     print(df.info())
#     s=predict_from_dataframe_safe(df)
#     print(s)
