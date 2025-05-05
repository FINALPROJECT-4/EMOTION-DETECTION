print("Starting email_alert.py...")  # ← Add this line first!

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    from_email = "batch2025project@gmail.com"
    app_password = "lspqbqkwabynofxo"  # Use the app password from Google

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    print("Running test email...")
    send_email(
        subject="Test Email",
        body="This is a test to verify email functionality.",
        to_email="batch2025project@gmail.com"
    )
