
import os
import sys
import uuid
import io
import base64
import secrets
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session, abort
from werkzeug.utils import secure_filename
import traceback
# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.prediction_service import predict_default,predict_from_dataframe_safe,save_results_to_csv,save_results_to_pdf
from src.feature_engineering import preprocess_data
from src.single_feature_engineering import preprocess_single_row
# from src.visualization import visualize_data
# from src.utils import save_model_version_info
from src.train_model import train_and_save_model
import openpyxl
import xlrd
from flask_mail import Mail, Message
import urllib.parse

from src.prediction_service import (
    predict_default,
    predict_from_dataframe_safe,
    save_results_to_csv,
    save_results_to_pdf,
    save_batch_pdf,
    save_batch_txt
)
from src.utils.email_service import (
    init_mail,
    send_email_with_pdf,
    send_email_with_attachment,
    send_email_with_multiple_attachments
)
from src.data_analysis import (
    analyze_statistical_measures,
    check_duplicates,
    check_unique_values,
    generate_correlation_heatmap,
    analyze_payment_history,
    generate_data_quality_report
)
from src.feature_importance import (
    calculate_feature_importance_from_model,
    plot_feature_importance,
    generate_feature_importance_report,
    identify_top_predictive_features
)
from src.model_comparison import (
    compare_models_lazy_classifier,
    format_models_comparison,
    plot_models_comparison,
    get_model_recommendations
)

# Flask App Setup
app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

init_mail(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
REQUIRED_COLUMNS = 23
# mail config
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your_email@gmail.com'       # Replace with your email
# app.config['MAIL_PASSWORD'] = 'your_app_password_here'     # Use App Password, not your real password

# app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USER')
# app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASS')

# mail = Mail(app)

# Get absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Use this to point to downloads folder
DOWNLOADS_FOLDER = os.path.join(BASE_DIR, 'downloads')

# def send_email_with_pdf(to_email, subject, body, pdf_filename):
#     """Send email with PDF attachment for single prediction."""
#     try:
#         if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
#             print("⚠️ Email not configured. Set MAIL_USER and MAIL_PASS environment variables.")
#             return False
            
#         msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[to_email])
#         msg.body = body
#         msg.html = f"<p>{body.replace(chr(10), '<br>')}</p>"
        
#         # Attach PDF if it exists
#         pdf_path = os.path.join("app", "static", pdf_filename)
#         if os.path.exists(pdf_path):
#             with open(pdf_path, 'rb') as f:
#                 msg.attach(filename=f"{pdf_filename}", content_type="application/pdf", data=f.read())
        
#         # Attach TXT if it exists
#         txt_filename = pdf_filename.replace('.pdf', '.txt')
#         txt_path = os.path.join("app", "static", txt_filename)
#         if os.path.exists(txt_path):
#             with open(txt_path, 'r', encoding='utf-8') as f:
#                 msg.attach(filename=f"{txt_filename}", content_type="text/plain", data=f.read())
        
#         mail.send(msg)
#         print(f"✅ Email sent to {to_email}")
#         return True
#     except Exception as e:
#         print(f"❌ Email error: {str(e)}")
#         return False


# def send_email_with_attachment(to_email, subject, body, file_path, filename):
    # """Generic function to send email with file attachment."""
    # try:
    #     if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
    #         print("⚠️ Email not configured. Set MAIL_USER and MAIL_PASS environment variables.")
    #         return False
            
    #     msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[to_email])
    #     msg.body = body
    #     msg.html = f"<p>{body.replace(chr(10), '<br>')}</p>"
        
    #     if os.path.exists(file_path):
    #         with open(file_path, 'rb') as f:
    #             content_type = 'text/csv' if filename.endswith('.csv') else 'application/pdf'
    #             msg.attach(filename=filename, content_type=content_type, data=f.read())
        
    #     mail.send(msg)
    #     print(f"✅ Email sent to {to_email}")
    #     return True
    # except Exception as e:
    #     print(f"❌ Email error: {str(e)}")
    #     return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx'}

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     result = None
#     if request.method == 'POST':
#         try:
#             # Collect input using friendly names from the updated HTML
#             fields = [
#                 'limit_balance', 'sex', 'education', 'marital_status', 'age', 'Credit Card Type', 'Credit Card Limit Utilization','Credit Card Usage'
#                 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
#                 'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun', 'bill_amt_may', 'bill_amt_apr',
#                 'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun', 'pay_amt_may', 'pay_amt_apr'
#             ]            
#             input_data = []
#             print(request.form)  # Debugging line to check form data
#             for field in fields:
#                 value = request.form.get(field)
#                 if value is None or value.strip() == "":
#                     flash(f"⚠️ Missing value for '{field}'. Please fill it in.", "danger")
#                     return render_template('index.html')

#                 missing = [field if not request.form.get(field) else None for field in fields]
#                 if missing:
#                     missing_str = ", ".join(fields.get(f, f) for f in missing)
#                     flash(f"⚠️ Please fill in: {missing_str}", "danger")
#                     return render_template('index.html')
#                 try:
#                     input_data.append(float(value))
#                 except ValueError:
#                     flash(f"❌ Invalid input for '{field}': {value}", "danger")
#                     return render_template('index.html')

#             pred, prob = predict_default(input_data)
#             result = f"Default: {'Yes' if pred == 1 else 'No'} (Probability: {prob:.2f}%)"
#         except Exception as e:
#             result = f"Invalid input. Please check your data. Error: {str(e)}"

#     return render_template('index.html', result=result)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    shap_filename = None
    pdf_filename = None
    txt_filename = None
    submitted_values = {}
    show_scroll_notice = False
    if request.method == 'POST':
        submitted_values = request.form.to_dict()
        fields = [
            'limit_balance', 'Gender', 'education', 'age', 
            #'marital_status', 'Credit Card Type', 'Credit Card Limit Utilization', 'Credit Card Usage',
            'PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun', 'bill_amt_may', 'bill_amt_apr',
            'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun', 'pay_amt_may', 'pay_amt_mar'
        ]
        optional_field = ['Gender', 'education','age']
        print(request.form)  # Debugging line to check form data
        print("========== FORM DATA ==========")
        print(submitted_values)
        print("EMAIL RECEIVED:", repr(submitted_values.get('email')))
        print("================================")

        # Collect input data    
        input_data = []
        missing_fields = []
        # Check for missing fields and convert to float
        for field in fields:
            value = request.form.get(field)
            if value is None or value.strip() == "":
                missing_fields.append(field)
                continue
            try:
                input_data.append(float(value))
            except ValueError:
                flash(f"❌ Invalid value for '{field}': {value}", "danger")
                return render_template('index.html')

        if missing_fields:
            flash(f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}", "danger")
            return render_template('index.html')
        try:
            # You already extract input_data from the form

            # Preprocess the input data
            # Processed_input = preprocess_single_row(input_data)
            # Step 2: Get form data and convert to dict
            # form_data = request.form.to_dict()

            # Step 3: Filter and convert to DataFrame
            # (Make sure only expected features are included, and values are converted to numeric)
            # filtered_data = {key: float(form_data[key]) for key in fields}
            # df = pd.DataFrame([filtered_data])
            df = request.form
            user_values = {k: v for k, v in submitted_values.items() if k not in {"output_format", "email", "email_report"}}
            print("RECEIVED DATA:")
            pred, prob, shap_filename, pdf_filename, txt_filename = predict_default(df, user_values=user_values)
            result = f"Default: {'Yes' if pred == 1 else 'No'} (Probability: {prob:.2f}%)"
            show_scroll_notice = True
            print("RESULT GENERATED.")
            
            # Send reports via email if email is provided
            email = submitted_values.get('email', '').strip()
            if email:
                email_subject = "Credit Card Default Prediction Report"
                email_body = f"""
Dear User,

Your credit card default prediction analysis is complete!

PREDICTION RESULT: {'HIGH RISK (Default)' if pred == 1 else 'LOW RISK (No Default)'}
Probability of Default: {prob:.2f}%

Your detailed reports (PDF and TXT formats) are attached to this email.

The PDF report includes:
- Your prediction result with confidence level
- Your provided information for reference
- How the result was calculated
- Feature contribution analysis (SHAP plot showing which features influenced the prediction)

The TXT report contains the same information in plain text format.

For questions or concerns, please contact support.

Best regards,
Credit Card Default Prediction System
"""
                sent = send_email_with_pdf(
                    to_email=email,
                    subject=email_subject,
                    body=email_body,
                    pdf_filename=pdf_filename
                )
                if sent:
                    flash(f"✅ Report sent successfully to {email}!", "success")
                else:
                    flash("⚠️ Could not send email. Email configuration may not be set up properly.", "warning")
            
            return render_template('index.html', result=result, shap_path=shap_filename, pdf_path=pdf_filename, txt_path=txt_filename, show_scroll_notice=show_scroll_notice, submitted_values=submitted_values)
        except Exception as e:
            result = f"Invalid input. Please check your data. Error: {str(e)}"
            flash(result, "danger")
            # if not shap_filename:
            #     flash("SHAP plot could not be generated.", "warning")
            #     shap_filename = None
            #     flash(result, "danger")

    return render_template('index.html', submitted_values=submitted_values)

# Route 2: Upload form
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('Invalid file.', 'danger')
            return redirect(url_for('batch'))
        

        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)

        session['uploaded_file'] = filepath
        flash('File uploaded successfully!', 'success')
        return redirect(url_for('validate'))
        # if session.get('processed_file'):
        #     flash('File already processed. Continue for prediction.', 'info')
        #     return redirect(url_for('run_prediction'))

    return render_template('batch.html')
    # return render_template(
    #     'batch.html',
    #     prediction_done=True,
    #     run_id=session.get('run_id'),
    #     download_link=generated_filename,
    #     email_sent=True
    # )#need to xcheck this tomo


@app.route('/validate', methods=['GET', 'POST'])
def validate():
    file_path = session.get('uploaded_file')
    if not file_path:
        flash('No file found. Please upload again.', 'danger')
        return redirect(url_for('batch'))
        # Determine extension

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.xlsx':
        df = pd.read_excel(file_path, engine='openpyxl')
    elif ext == '.xls':
        df = pd.read_excel(file_path, engine='xlrd')
    else:
        flash("Unsupported file format.", "danger")
        return redirect(url_for('batch'))

    # if not set(expected_features).issubset(df.columns):
    #     flash('File missing required columns.', 'danger')
    #     return redirect(url_for('batch'))

    try:
        flash('File uploaded. Running validation and preprocessing...', 'info')
        clean_data, run_dir = preprocess_data(df)
        os.makedirs(run_dir, exist_ok=True)  # ensure directory exists
        print(f"run_dir = {run_dir}")
        encoded_run_dir = urllib.parse.quote(run_dir)
        session["uploadedfile"] = df.shape
        session['processed_file'] = clean_data
        session['run_dir'] = run_dir
        # In your Flask route (run_prediction), do:
        # flash('File validated and preprocessed successfully.', 'success')
        # return redirect(url_for('run_prediction'))
        flash('File validated and preprocessed successfully. Click below to run prediction.', 'success')
        return render_template('batch.html', show_run_prediction_button=True)

    except Exception as e:
        flash(f'Preprocessing error: {str(e)}', 'danger')
        return redirect(url_for('batch'))

@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    try:
        run_dir = session["run_dir"]
        format_selected = request.form.get('output_format')
        email = request.form.get('email')
        filepath = session.get('processed_file')
        original_df = session.get("uploadedfile")
        output_format = request.form.get('output_format')
        if not filepath or not output_format:
            flash("File or format not provided.", "danger")
            return redirect(url_for('batch'))

        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        results = predict_from_dataframe_safe(df)
        original_shape = original_df
        processed_shape = df.shape
        # results = [
        #     {
        #         'Prediction': 'Default' if predict_default(row.tolist())[0] == 1 else 'No Default',
        #         'Probability (%)': round(predict_default(row.tolist())[1] * 100, 2)
        #     }
        #     for _, row in df.iterrows()
        # ]
    #     def predict_default(input_data):
    # if isinstance(input_data, pd.Series):
    #     input_data = pd.DataFrame([input_data])
    # elif isinstance(input_data, list):
    #     raise ValueError("Expected DataFrame or Series, got list")

    # # Proceed as normal

        # df_result = pd.DataFrame(results)
        # os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # output_path = os.path.join(OUTPUT_FOLDER, run_dir)
        # # df_result.to_csv(output_path, index=False)

        # Add email send logic here if needed
        # In your Flask route (run_prediction), do:
        # os.makedirs(run_dir, exist_ok=True)  # ensure directory exists
         # Save results always into the "downloads" folder
        # BASE_DIR = r"C:\Users\masra\Desktop\Project\credit-card-prediction"
        # output_dir = os.path.join(BASE_DIR, "downloads")
        os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
        output_filename = f"predictions_result.{output_format}"
        output_path = os.path.join(DOWNLOADS_FOLDER, output_filename)

        # Save to selected format
        if output_format == 'csv':
            filename = save_results_to_csv(results, output_path)
        elif output_format == 'pdf':
            filename = save_results_to_pdf(results, output_path)
        else:
            flash("Invalid output format selected.")
            return redirect('/batch')
        # do this later
        # append_predictions_to_dataframe(df,results)
        # # Offer download
        # return send_file(output_file, as_attachment=True)
        flash('Prediction completed , File is ready.', 'success')
        
        # Send results via email if provided - WITH ALL ATTACHMENTS
        email_sent = False
        if email:
            email = email.strip()
            if email:  # Double check email is not empty
                # Collect ALL files to attach
                files_to_attach = {}
                
                # 1. Add predictions results file (CSV or PDF)
                predictions_file = os.path.join(DOWNLOADS_FOLDER, filename)
                if os.path.exists(predictions_file):
                    files_to_attach[predictions_file] = f"batch_predictions_results.{output_format}"
                
                # 2. Add all files from run_dir if it exists
                run_dir = session.get('run_dir')
                if run_dir and os.path.exists(run_dir):
                    print(f"🔍 Collecting files from run_dir: {run_dir}")
                    
                    for filename_in_dir in os.listdir(run_dir):
                        file_path = os.path.join(run_dir, filename_in_dir)
                        
                        # Skip directories and pickle files
                        if os.path.isdir(file_path) or filename_in_dir.endswith('.pkl'):
                            continue
                        
                        # Add file with renamed display name for clarity
                        if 'preprocessed_data' in filename_in_dir:
                            display_name = "preprocessed_data.csv"
                        elif 'model_metrics' in filename_in_dir:
                            display_name = "model_metrics_report.txt"
                        elif 'roc_curve' in filename_in_dir:
                            display_name = "roc_curve.png"
                        elif 'precision_recall' in filename_in_dir:
                            display_name = "precision_recall_curve.png"
                        elif 'metrics_report' in filename_in_dir:
                            display_name = "detailed_metrics_report.html"
                        else:
                            display_name = filename_in_dir
                        
                        files_to_attach[file_path] = display_name
                        print(f"  ✅ Added: {display_name}")
                
                # 3. Prepare comprehensive email
                email_subject = "Credit Card Batch Prediction - Complete Results & Reports"
                email_body = f"""
Dear User,

Your batch prediction analysis is complete! 

BATCH PROCESSING SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Original Records Uploaded: {original_shape[0]}
• Processed Records: {processed_shape[0]}
• Output Format: {output_format.upper()}
• Total Predictions: {len(results)}

ATTACHMENTS INCLUDED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PREDICTIONS & DATA:
  • batch_predictions_results.{output_format} - All {len(results)} predictions with probabilities
  • preprocessed_data.csv - Your processed data (all {processed_shape[0]} records)

📈 MODEL PERFORMANCE REPORTS:
  • model_metrics_report.txt - Confusion matrix and classification metrics
  • roc_curve.png - ROC (Receiver Operating Characteristic) curve
  • precision_recall_curve.png - Precision-Recall curve
  • detailed_metrics_report.html - Interactive metrics dashboard (open in browser)

PREDICTION RESULTS BREAKDOWN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
                
                # Calculate statistics
                defaults_count = sum(1 for r in results if r.get('Prediction') == 'Default')
                no_defaults_count = len(results) - defaults_count
                default_rate = (defaults_count / len(results) * 100) if results else 0
                
                email_body += f"""
  • Default Risk Records: {defaults_count} ({default_rate:.2f}%)
  • No Default Risk Records: {no_defaults_count} ({100-default_rate:.2f}%)

HOW TO USE THESE FILES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. batch_predictions_results.{output_format}: Open with Excel/spreadsheet app to view all predictions
2. preprocessed_data.csv: View your cleaned and normalized data
3. ROC & PR Curves: Use to assess model performance
4. Metrics Report: Detailed performance indicators (Accuracy, Precision, Recall, F1-Score)
5. HTML Dashboard: Open in web browser for interactive visualization

NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Review the predictions in the attached CSV/PDF
✅ Check the model performance in the attached reports
✅ Contact support if you have questions about the results
✅ Use these insights for your credit risk assessment decisions

For questions or concerns, please contact support.

Best regards,
Credit Card Default Prediction System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@2026 Credit Card Default Prediction System | By Masrath Unnissa
"""
                
                # 4. Send email with ALL attachments
                if files_to_attach:
                    sent = send_email_with_multiple_attachments(
                        to_email=email,
                        subject=email_subject,
                        body=email_body,
                        file_paths=files_to_attach
                    )
                    if sent:
                        email_sent = True
                        num_attachments = len(files_to_attach)
                        flash(f"✅ Complete results sent to {email} with {num_attachments} attachment(s)!", "success")
                    else:
                        flash("⚠️ Could not send email. Email configuration may not be set up properly.", "warning")
                else:
                    flash("⚠️ No files available to email.", "warning")
        
        # return render_template('batch.html',  prediction_done=True, filename=os.path.basename(output_path), run_id=encoded_run_dir, filename='prediction_result.csv')#make batch have result and form select html changes
        return render_template(
            'batch.html',
            prediction_done=True,
            run_dir=run_dir,
            filename=filename,
            original_rows=original_shape[0],
            original_cols=original_shape[1],
            processed_rows=processed_shape[0],
            processed_cols=processed_shape[1],
            sample_data=df.head().to_html(classes='table table-striped'),
            # download_link=filename,
            email_sent=email_sent,
            email=email
        ) #check tomo

    except Exception as e:
        traceback.print_exc()
        flash(f'Preprocessing error: {str(e)}', 'danger')
        return redirect(url_for('batch'))


@app.route('/view_roc')
def view_roc():
    encoded_path = request.args.get('run_dir')
    if not encoded_path:
        abort(400, "Missing run_dir")

    run_dir = urllib.parse.unquote(encoded_path)  # Decode it
    image_path = os.path.join(run_dir, 'roc_curve.png')

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        abort(404, "ROC image not found")

    return redirect(url_for("batch")) 

@app.route('/view_pr')
def view_pr():
    encoded_path = request.args.get('run_dir')
    if not encoded_path:
        abort(400, "Missing run_dir")

    run_dir = urllib.parse.unquote(encoded_path)  # Decode it
    image_path = os.path.join(run_dir, 'precision_recall_curve.png')
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        flash("PR curve not found.", "danger")
        return redirect(url_for("batch"))
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(DOWNLOADS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash(f"File '{filename}' not found.")
        return redirect('/batch')

# @app.route('/download/<filename>')
# def download_file(filename):
#     try:
#         # Sanitize the filename to prevent path traversal
#         safe_filename = secure_filename(filename)

#         # Build the safe full path
#         full_path = os.path.join(OUTPUT_FOLDER, safe_filename)

#         # Check if the file actually exists
#         if not os.path.exists(full_path):
#             flash(f"File '{safe_filename}' not found.", 'danger')
#             return redirect(url_for('batch'))

#         # Log download event (optional)
#         app.logger.info(f"User downloaded file: {safe_filename}")

#         # Send file as download
#         return send_file(full_path, as_attachment=True)

#     except Exception as e:
#         app.logger.error(f"Download error: {str(e)}")
#         flash("An error occurred while preparing your download.", 'danger')
#         return redirect(url_for('batch'))
# Route to view and download model metrics
@app.route('/view_metrics')
def view_metrics():
    """Display metrics report as HTML in browser."""
    from src.prediction_service import generate_html_metrics_report
    
    encoded_path = request.args.get('run_dir')
    if not encoded_path:
        abort(400, "Missing run_dir")

    run_dir = urllib.parse.unquote(encoded_path)  # Decode it
    
    try:
        # Find and parse the metrics file
        metrics_file = None
        for file in os.listdir(run_dir):
            if file.startswith("model_metrics") and file.endswith(".txt"):
                metrics_file = os.path.join(run_dir, file)
                break
        
        if not metrics_file:
            flash("Metrics file not found.", "danger")
            return redirect(url_for("batch"))
        
        # Generate HTML report
        html_output_path = os.path.join(run_dir, "metrics_report.html")
        generate_html_metrics_report(metrics_file, html_output_path)
        
        # Read and display the HTML
        with open(html_output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("batch"))


@app.route('/download/metrics')
def download_metrics():
    encoded_path = request.args.get('run_dir')
    if not encoded_path:
        abort(400, "Missing run_dir")

    run_dir = urllib.parse.unquote(encoded_path)  # Decode it
 
    try:
        # Search for the metrics file
        for file in os.listdir(run_dir):
            if file.startswith("model_metrics") and file.endswith(".txt"):
                return send_file(
                    os.path.join(run_dir, file),
                    as_attachment=True
                )
        flash("Metrics file not found.", "danger")
        return redirect(url_for("batch"))
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("batch"))

@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Train model and perform comprehensive analysis including:
    - Statistical analysis
    - Data quality checks
    - Correlation analysis
    - Payment history analysis
    - Feature importance
    - Model comparison with LazyClassifier
    """
    analysis_results = {}
    data_loaded = False
    
    if request.method == 'POST':
        try:
            uploaded_file = request.files.get('file')
            
            if not uploaded_file or not uploaded_file.filename:
                flash('Please upload a file.', 'danger')
                return redirect(url_for('train'))
            
            if not uploaded_file.filename.endswith(('.csv', '.xls', '.xlsx')):
                flash('Please upload a valid file (.csv or .xls/.xlsx)', 'danger')
                return redirect(url_for('train'))
            
            print("📊 Loading data for analysis...")
            
            # Load data
            try:
                if uploaded_file.filename.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                flash(f'Error loading file: {str(e)}', 'danger')
                return redirect(url_for('train'))
            
            print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            data_loaded = True

            # Keep analysis column names compatible with the training pipeline.
            analysis_df = df.copy()
            analysis_df.columns = [str(column).strip() for column in analysis_df.columns]
            canonical_columns = [
                'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                'PAY_AMT3', 'PAY_AMT4'
            ]
            column_aliases = {
                ''.join(character for character in column.lower() if character.isalnum()): column
                for column in canonical_columns
            }
            normalized_columns = []
            for column in analysis_df.columns:
                key = ''.join(character for character in column.lower() if character.isalnum())
                normalized_columns.append(column_aliases.get(key, column.replace(' ', '_')))
            analysis_df.columns = normalized_columns
            if 'PAY_0' not in analysis_df.columns and len(analysis_df.columns) >= len(canonical_columns):
                analysis_df.columns = canonical_columns + list(analysis_df.columns[len(canonical_columns):])
            for column in analysis_df.columns:
                converted = pd.to_numeric(analysis_df[column], errors='coerce')
                if converted.notna().any():
                    analysis_df[column] = converted

            target_aliases = {
                'default_payment_next_month': 'default',
                'defaultpaymentnextmonth': 'default',
                'default.payment.next.month': 'default',
                'default': 'default'
            }
            analysis_df.rename(
                columns={column: target_aliases[column] for column in analysis_df.columns if column in target_aliases},
                inplace=True
            )
            required_features = {
                'LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4'
            }
            if 'default' not in analysis_df.columns and required_features.issubset(analysis_df.columns):
                serious_delay = analysis_df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].ge(2).any(axis=1)
                bill_total = analysis_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
                pay_total = analysis_df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4']].mean(axis=1)
                low_payment_ratio = (pay_total / (bill_total + 1e-6)) < 0.2
                analysis_df['default'] = (
                    serious_delay |
                    low_payment_ratio |
                    ((analysis_df['LIMIT_BAL'] < 100000) & (analysis_df['PAY_0'] > 0))
                ).astype(int)
            
            # ====================================
            # 1. DATA QUALITY ANALYSIS
            # ====================================
            print("\n📋 Starting data quality analysis...")
            quality_report = generate_data_quality_report(analysis_df)
            analysis_results['quality_report'] = quality_report
            
            # ====================================
            # 2. STATISTICAL MEASURES
            # ====================================
            print("📈 Calculating statistical measures...")
            stats_result = analyze_statistical_measures(analysis_df)
            analysis_results['statistics'] = stats_result
            
            # ====================================
            # 3. DUPLICATE CHECK
            # ====================================
            print("🔍 Checking duplicates...")
            duplicates = check_duplicates(analysis_df)
            analysis_results['duplicates'] = duplicates
            
            # ====================================
            # 4. UNIQUE VALUES CHECK
            # ====================================
            print("🔢 Checking unique values...")
            unique_values = check_unique_values(analysis_df)
            analysis_results['unique_values'] = unique_values
            
            # ====================================
            # 5. CORRELATION HEATMAP
            # ====================================
            print("🔗 Generating correlation heatmap...")
            heatmap_result = generate_correlation_heatmap(analysis_df)
            analysis_results['heatmap'] = heatmap_result
            
            # ====================================
            # 6. PAYMENT HISTORY ANALYSIS
            # ====================================
            print("💳 Analyzing payment history...")
            payment_analysis = analyze_payment_history(analysis_df, target_column='default')
            analysis_results['payment_analysis'] = payment_analysis
            
            # ====================================
            # 7. FEATURE IMPORTANCE (if model available)
            # ====================================
            print("⭐ Analyzing feature importance...")
            try:
                from src.prediction_service import model, scaler, fields
                
                # Prepare data for model
                feature_count = getattr(model, 'n_features_in_', len(fields))
                feature_names = list(fields[:feature_count])
                if len(feature_names) < feature_count:
                    feature_names.extend(
                        f'feature_{index}'
                        for index in range(len(feature_names), feature_count)
                    )
                importance_dict = calculate_feature_importance_from_model(model, feature_names)
                if importance_dict:
                    importance_plot = plot_feature_importance(importance_dict, top_n=15)
                    importance_report = generate_feature_importance_report(importance_dict)
                    top_features = identify_top_predictive_features(importance_dict, threshold=0.8)
                    analysis_results['feature_importance'] = {
                        'plot': importance_plot,
                        'report': importance_report,
                        'top_features': top_features
                    }
            except Exception as e:
                print(f"⚠️ Feature importance calculation skipped: {str(e)}")
            
            # ====================================
            # 8. MODEL COMPARISON (LazyClassifier)
            # ====================================
            print("\n🤖 Starting model comparison...")
            try:
                from sklearn.model_selection import train_test_split
                
                # Prepare data
                numeric_df = analysis_df.select_dtypes(include=[np.number]).copy()
                if len(numeric_df) > 1:
                    # Find target column (last column or 'default.payment.next.month')
                    target_col = None
                    if 'default' in numeric_df.columns:
                        target_col = 'default'
                    else:
                        # Use last column as target
                        target_col = numeric_df.columns[-1]
                    
                    if target_col in numeric_df.columns:
                        X = numeric_df.drop(target_col, axis=1)
                        y = numeric_df[target_col]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Compare models
                        comparison_result = compare_models_lazy_classifier(
                            X_train, X_test, y_train, y_test
                        )
                        
                        if comparison_result:
                            # Format results
                            formatted = format_models_comparison(comparison_result)
                            if formatted:
                                # Generate visualizations
                                plot_result = plot_models_comparison(formatted, metric='Accuracy', top_n=15)
                                recommendations = get_model_recommendations(formatted)
                                
                                analysis_results['model_comparison'] = {
                                    'formatted': formatted,
                                    'plot': plot_result,
                                    'recommendations': recommendations
                                }
                                
                                print(f"✅ Model comparison completed: {formatted['models_count']} models tested")
            except ImportError:
                print("⚠️ LazyClassifier not available. Install: pip install lazypredict")
                flash('⚠️ LazyClassifier not installed. To use model comparison, run: pip install lazypredict', 'info')
            except Exception as e:
                print(f"⚠️ Model comparison skipped: {str(e)}")
            
            # ====================================
            # 9. TRAIN YOUR MAIN MODEL
            # ====================================
            print("\n🚀 Training main model...")
            try:
                accuracy, report, model_path, scaler_path = train_and_save_model(analysis_df)
                analysis_results['model_training'] = {
                    'model_path': model_path,
                    'accuracy': accuracy
                }
                flash(f'✅ Model trained successfully! Accuracy: {accuracy:.2f}%', 'success')
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'danger')
            
            print("\n✅ All analyses completed!")
            
            # Return results to template
            return render_template(
                'train.html',
                data_loaded=True,
                analysis_results=analysis_results
            )
        
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            traceback.print_exc()
            flash(f'Error during analysis: {str(e)}', 'danger')
            return redirect(url_for('train'))
    
    return render_template('train.html', data_loaded=False, analysis_results={})

@app.route('/version')
def version():
    import json
    try:
        with open('models/model_versions.json', 'r') as f:
            versions = json.load(f)
    except FileNotFoundError:
        versions = {}

    return render_template('version.html', versions=versions)

@app.route('/visual', methods=['GET', 'POST'])
def visual():
    graph_url = None
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        filename = uploaded_file.filename.lower() if uploaded_file else ''
        if not uploaded_file or not filename:
            flash('Please select a CSV or Excel file.', 'danger')
            return redirect(url_for('visual'))
        if not filename.endswith(('.csv', '.xls', '.xlsx')):
            flash('Only CSV and Excel files are supported for visualization.', 'danger')
            return redirect(url_for('visual'))

        try:
            df = pd.read_csv(uploaded_file) if filename.endswith('.csv') else pd.read_excel(uploaded_file)
            results = predict_from_dataframe_safe(df)
            probabilities = [
                float(str(result['Probability']).rstrip('%'))
                for result in results
                if result.get('Probability') is not None
            ]
            if not probabilities:
                raise ValueError('No prediction probabilities were produced.')

            plt.figure(figsize=(8, 6))
            plt.hist(probabilities, bins=10, color='skyblue', edgecolor='black')
            plt.xlabel('Probability of Default (%)')
            plt.ylabel('Number of Customers')
            plt.title('Default Risk Distribution')
            plt.tight_layout()

            image = io.BytesIO()
            plt.savefig(image, format='png', dpi=100)
            plt.close()
            graph_url = f"data:image/png;base64,{base64.b64encode(image.getvalue()).decode()}"
        except Exception as error:
            flash(f'Unable to generate risk visualization: {error}', 'danger')

    return render_template('visuals.html', graph_url=graph_url)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
