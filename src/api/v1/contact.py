"""
Contact form API endpoint with Gmail integration
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

def send_email_via_gmail(name: str, email: str, subject: str, message: str) -> bool:
    """Send email using Gmail SMTP"""
    try:
        # Gmail SMTP configuration
        gmail_user = os.getenv('GMAIL_USER', 'info@agenttradr.com')
        gmail_password = os.getenv('GMAIL_APP_PASSWORD', '')  # App-specific password
        
        if not gmail_password:
            logger.error("Gmail app password not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = gmail_user  # Send to yourself
        msg['Subject'] = f"Contact Form: {subject}"
        
        # Email body
        body = f"""
New contact form submission from AgentTRADR website:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This email was sent from the AgentTRADR contact form.
        """.strip()
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable security
        server.login(gmail_user, gmail_password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(gmail_user, gmail_user, text)
        server.quit()
        
        logger.info(f"Contact form email sent successfully from {email}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

@router.post("/contact")
async def submit_contact_form(contact_data: ContactForm) -> Dict[str, Any]:
    """Handle contact form submission"""
    try:
        # Send email
        success = send_email_via_gmail(
            name=contact_data.name,
            email=contact_data.email,
            subject=contact_data.subject,
            message=contact_data.message
        )
        
        if success:
            return {
                "success": True,
                "message": "Thank you for your message! We'll get back to you soon."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
            
    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")