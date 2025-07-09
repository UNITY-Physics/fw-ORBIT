from datetime import datetime
from app.main import reporter
from util.email import send_email_with_csv

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from email.utils import formataddr
import os


# Example usage
csv_file_path = f"logs/usage-report-{datetime.now().strftime('%Y-%m-%d')}.csv"

# Define email details
sender_email = "kclcns@gmail.com"  
sender_name = "KCL UNITY"
recipient_email = "hajer.karoui@kcl.ac.uk"
subject = "Flywheel UNITY Data Missingness"
body = "Please find the attached csv to fill out and upload via the information sync gear."
smtp_server = "smtp.gmail.com"  # e.g., "smtp.gmail.com" for Gmail
smtp_port = 587  # TLS port for most SMTP servers
smtp_username = "kclcns@gmail.com"  # Your email username
smtp_password = ""  # Your email password

# Get the usage report
reporter(csv_file_path)

# Send the email with the CSV attachment
send_email_with_csv(sender_email, sender_name, recipient_email, subject, body, csv_file_path, smtp_server, smtp_port, smtp_username, smtp_password)


# Function to send an email with a CSV attachment
def send_email_with_csv(sender_email, sender_name, recipient_email, subject, body, file_path, smtp_server, smtp_port, smtp_username, smtp_password):
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = formataddr((sender_name, sender_email))
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Add the email body
    msg.attach(MIMEText(body, 'plain'))

    # Attach the CSV file
    with open(file_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(file_path)}",
        )
        msg.attach(part)

    # Send the email via SMTP
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Error sending email: {e}")