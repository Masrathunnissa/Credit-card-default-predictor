import os
from flask_mail import Mail, Message

# --------------------------------------------------
# Flask-Mail instance
# --------------------------------------------------

mail = Mail()


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "app",
    "static"
)

ENV_FILE = os.path.join(BASE_DIR, ".env")


# --------------------------------------------------
# Load environment variables from .env file
# --------------------------------------------------

def load_env_file(env_path=ENV_FILE):
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file
    
    Returns:
        dict: Dictionary of environment variables
    """
    env_vars = {}
    
    if not os.path.exists(env_path):
        print(f"⚠️ .env file not found at: {env_path}")
        return env_vars
    
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    env_vars[key] = value
        
        print(f"✅ Loaded .env file from: {env_path}")
    
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
    
    return env_vars


# Load environment variables
_env_vars = load_env_file()


def get_env(key, default=None):
    """
    Get environment variable from .env file or system environment.
    
    Args:
        key: Variable name
        default: Default value if not found
    
    Returns:
        Variable value or default
    """
    # First check .env file
    if key in _env_vars and _env_vars[key]:
        return _env_vars[key]
    
    # Then check system environment variables
    value = os.environ.get(key)
    if value:
        return value
    
    # Return default
    return default


# --------------------------------------------------
# Initialize mail
# --------------------------------------------------

def init_mail(app):
    """
    Initialize Flask-Mail with the Flask application.
    Reads configuration from .env file or system environment variables.
    """

    app.config["MAIL_SERVER"] = get_env("MAIL_SERVER", "smtp.gmail.com")
    app.config["MAIL_PORT"] = int(get_env("MAIL_PORT", "587"))
    app.config["MAIL_USE_TLS"] = get_env("MAIL_USE_TLS", "True").lower() == "true"
    app.config["MAIL_USERNAME"] = get_env("MAIL_USER")
    app.config["MAIL_PASSWORD"] = get_env("MAIL_PASS")

    mail.init_app(app)

    print("📧 Mail service initialized")

    if not app.config["MAIL_USERNAME"]:
        print("⚠️ MAIL_USER not configured in .env file or environment variables.")
        print(f"   📝 Please edit: {ENV_FILE}")

    if not app.config["MAIL_PASSWORD"]:
        print("⚠️ MAIL_PASS not configured in .env file or environment variables.")
        print(f"   📝 Please edit: {ENV_FILE}")


# --------------------------------------------------
# Send PDF + TXT
# --------------------------------------------------

def send_email_with_pdf(
    to_email,
    subject,
    body,
    pdf_filename,
    txt_filename=None
):
    """
    Send an email with PDF and optional TXT attachments.

    Files are expected inside:
        app/static/
    """

    try:
        mail_username = get_env("MAIL_USER")
        mail_password = get_env("MAIL_PASS")

        if not mail_username or not mail_password:
            print(
                "❌ Email not configured. "
                f"Please configure MAIL_USER and MAIL_PASS in: {ENV_FILE}"
            )
            return False

        if not to_email:
            print("❌ Recipient email is empty.")
            return False

        msg = Message(
            subject=subject,
            sender=mail_username,
            recipients=[to_email]
        )

        msg.body = body
        msg.html = body.replace("\n", "<br>")

        # ------------------------------------------
        # PDF
        # ------------------------------------------

        pdf_path = os.path.join(
            STATIC_DIR,
            pdf_filename
        )

        if os.path.exists(pdf_path):

            with open(pdf_path, "rb") as f:
                msg.attach(
                    filename=os.path.basename(pdf_filename),
                    content_type="application/pdf",
                    data=f.read()
                )

            print(f"📎 PDF attached: {pdf_path}")

        else:
            print(f"⚠️ PDF not found: {pdf_path}")

        # ------------------------------------------
        # TXT
        # ------------------------------------------

        if txt_filename is None:
            txt_filename = pdf_filename.replace(
                ".pdf",
                ".txt"
            )

        txt_path = os.path.join(
            STATIC_DIR,
            txt_filename
        )

        if os.path.exists(txt_path):

            with open(txt_path, "rb") as f:
                msg.attach(
                    filename=os.path.basename(txt_filename),
                    content_type="text/plain",
                    data=f.read()
                )

            print(f"📎 TXT attached: {txt_path}")

        else:
            print(f"⚠️ TXT not found: {txt_path}")

        # ------------------------------------------
        # Send
        # ------------------------------------------

        mail.send(msg)

        print(
            f"✅ Email successfully sent to {to_email}"
        )

        return True

    except Exception as e:

        print(
            f"❌ Email sending failed: "
            f"{type(e).__name__}: {e}"
        )

        return False


# --------------------------------------------------
# Generic attachment email
# --------------------------------------------------

def send_email_with_attachment(
    to_email,
    subject,
    body,
    file_path,
    filename
):
    """
    Send an email with a generic file attachment.
    """

    try:
        mail_username = get_env("MAIL_USER")
        mail_password = get_env("MAIL_PASS")

        if not mail_username or not mail_password:
            print(
                "❌ Email not configured. "
                f"Please configure MAIL_USER and MAIL_PASS in: {ENV_FILE}"
            )
            return False

        if not to_email:
            print("❌ Recipient email is empty.")
            return False

        if not os.path.exists(file_path):
            print(
                f"❌ Attachment file not found: {file_path}"
            )
            return False

        msg = Message(
            subject=subject,
            sender=mail_username,
            recipients=[to_email]
        )

        msg.body = body
        msg.html = body.replace("\n", "<br>")

        # Determine content type
        if filename.lower().endswith(".csv"):
            content_type = "text/csv"

        elif filename.lower().endswith(".pdf"):
            content_type = "application/pdf"

        elif filename.lower().endswith(".txt"):
            content_type = "text/plain"

        else:
            content_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            msg.attach(
                filename=os.path.basename(filename),
                content_type=content_type,
                data=f.read()
            )

        print(f"📎 Attachment added: {file_path}")

        mail.send(msg)

        print(
            f"✅ Email successfully sent to {to_email}"
        )

        return True

    except Exception as e:

        print(
            f"❌ Email sending failed: "
            f"{type(e).__name__}: {e}"
        )

        return False


# --------------------------------------------------
# Send email with multiple file attachments
# --------------------------------------------------

def send_email_with_multiple_attachments(
    to_email,
    subject,
    body,
    file_paths
):
    """
    Send an email with multiple file attachments.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body text
        file_paths: List of file paths to attach
                   Can be dict with {file_path: filename_for_email}
                   or just list of file paths
    
    Returns:
        True if successful, False otherwise
    """

    try:
        mail_username = get_env("MAIL_USER")
        mail_password = get_env("MAIL_PASS")

        if not mail_username or not mail_password:
            print(
                "❌ Email not configured. "
                f"Please configure MAIL_USER and MAIL_PASS in: {ENV_FILE}"
            )
            return False

        if not to_email:
            print("❌ Recipient email is empty.")
            return False

        msg = Message(
            subject=subject,
            sender=mail_username,
            recipients=[to_email]
        )

        msg.body = body
        msg.html = body.replace("\n", "<br>")

        # ------------------------------------------
        # Attach multiple files
        # ------------------------------------------

        attached_count = 0

        # Handle both list and dict formats
        if isinstance(file_paths, dict):
            files_to_attach = file_paths.items()
        else:
            files_to_attach = [(fp, os.path.basename(fp)) for fp in file_paths]

        for file_path, display_filename in files_to_attach:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            # Determine content type based on extension
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".csv":
                content_type = "text/csv"
            elif ext == ".pdf":
                content_type = "application/pdf"
            elif ext == ".txt":
                content_type = "text/plain"
            elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
                content_type = f"image/{ext.strip('.')}"
            elif ext == ".xlsx":
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif ext == ".xls":
                content_type = "application/vnd.ms-excel"
            else:
                content_type = "application/octet-stream"

            try:
                with open(file_path, "rb") as f:
                    msg.attach(
                        filename=display_filename,
                        content_type=content_type,
                        data=f.read()
                    )
                print(f"📎 Attached: {display_filename} ({os.path.getsize(file_path) / 1024:.1f} KB)")
                attached_count += 1
            except Exception as e:
                print(f"⚠️ Failed to attach {file_path}: {e}")
                continue

        if attached_count == 0:
            print("⚠️ No files were successfully attached.")

        # ------------------------------------------
        # Send
        # ------------------------------------------

        mail.send(msg)

        print(
            f"✅ Email successfully sent to {to_email} "
            f"with {attached_count} attachment(s)"
        )

        return True

    except Exception as e:

        print(
            f"❌ Email sending failed: "
            f"{type(e).__name__}: {e}"
        )

        return False
