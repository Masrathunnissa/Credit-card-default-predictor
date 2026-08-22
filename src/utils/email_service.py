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


# --------------------------------------------------
# Initialize mail
# --------------------------------------------------

def init_mail(app):
    """
    Initialize Flask-Mail with the Flask application.
    """

    app.config["MAIL_SERVER"] = "smtp.gmail.com"
    app.config["MAIL_PORT"] = 587
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USER")
    app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASS")

    mail.init_app(app)

    print("📧 Mail service initialized")

    if not app.config["MAIL_USERNAME"]:
        print("⚠️ MAIL_USER environment variable is not set.")

    if not app.config["MAIL_PASSWORD"]:
        print("⚠️ MAIL_PASS environment variable is not set.")


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
        mail_username = os.environ.get("MAIL_USER")
        mail_password = os.environ.get("MAIL_PASS")

        if not mail_username or not mail_password:
            print(
                "❌ Email not configured. "
                "Set MAIL_USER and MAIL_PASS environment variables."
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
        mail_username = os.environ.get("MAIL_USER")
        mail_password = os.environ.get("MAIL_PASS")

        if not mail_username or not mail_password:
            print(
                "❌ Email not configured. "
                "Set MAIL_USER and MAIL_PASS environment variables."
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
