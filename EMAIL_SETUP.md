# Email Setup Guide - Using .env Configuration File

## ✅ Quick Setup (3 Steps)

### Step 1: Add Gmail Credentials to .env File

1. Open the file: `.env` (in your project root)
2. Add your credentials:
   ```
   MAIL_USER=your-email@gmail.com
   MAIL_PASS=your-16-char-app-password
   ```

### Step 2: Get Gmail App Password

1. Go to: **https://myaccount.google.com/security**
2. Make sure **2-Step Verification is ON**
3. Search for **"App passwords"**
4. Select: Mail → Windows Computer
5. Click **Generate**
6. Copy the **16-character password** (remove spaces)
7. Paste into `.env` file as `MAIL_PASS`

### Step 3: Test It

Run your Flask app:
```powershell
python app/main.py
```

You should see:
```
📧 Mail service initialized
```

WITHOUT these warnings:
```
⚠️ MAIL_USER not configured in .env file
⚠️ MAIL_PASS not configured in .env file
```

---

## 📝 .env File Locations

**File path:** `C:\Users\phani\Desktop\MtechProject\Credit-card-default-predictor\.env`

**Content example:**
```
# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USER=john.doe@gmail.com
MAIL_PASS=abcdefghijklmnop
```

---

## 🔍 How It Works

The system now reads credentials from `.env` file in this order:
1. First checks `.env` file in project root
2. Falls back to system environment variables
3. Uses defaults if not found

No need to set Windows environment variables anymore! ✅

---

## ⚠️ Important Notes

- **DO NOT commit `.env` file to Git** (it's in .gitignore)
- Keep `.env.example` in Git as a template
- Never share your `.env` file - it has your password!
- Use **app-specific password** from Google, not your regular password
- The app-specific password is 16 characters (include all characters, remove spaces if copied)

---

## 🧪 Test Batch Email

1. Start Flask app: `python app/main.py`
2. Go to: http://localhost:5000/batch
3. Upload a CSV file
4. Run prediction
5. Enter your email address
6. Check your inbox for results! ✅

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Still seeing "not configured" warning | Restart Flask app after editing .env |
| Email not received | Check spam folder; verify 2FA enabled |
| "Authentication failed" | Make sure using app-specific password, not regular password |
| Empty email field in .env | Add your Gmail address and app password |

---

**That's it! Your email is now configured.** 🚀
