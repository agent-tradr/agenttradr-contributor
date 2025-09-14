"""
Simple contact form server for AgentTRADR
Serves static files and handles contact form submissions
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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
        gmail_password = os.getenv('GMAIL_APP_PASSWORD', '')
        
        if not gmail_password:
            logger.warning("Gmail app password not configured, email not sent")
            # For development, just log the message
            logger.info(f"Contact form submission: {name} ({email}): {subject} - {message}")
            return True  # Return True for development
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = gmail_user
        msg['Subject'] = f"AgentTRADR Contact: {subject}"
        
        body = f"""
New contact form submission from AgentTRADR website:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
Sent from AgentTRADR contact form
        """.strip()
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(gmail_user, gmail_user, text)
        server.quit()
        
        logger.info(f"Contact email sent successfully from {email}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

@app.post("/api/contact")
async def submit_contact_form(contact_data: ContactForm):
    """Handle contact form submission"""
    try:
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
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to send email"}
            )
            
    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"}
        )

# Mount static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/frontend/downloads", StaticFiles(directory="frontend/downloads"), name="downloads")

@app.get("/")
async def read_root():
    """Serve the main page"""
    return FileResponse("frontend/index.html")

@app.get("/frontend/index.html")
async def read_index():
    """Serve the index page"""
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)