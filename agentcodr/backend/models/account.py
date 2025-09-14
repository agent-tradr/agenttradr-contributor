"""Account model for AgentCODR backend."""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class Account(BaseModel):
    """User account model."""
    
    id: Optional[str] = Field(None, description="Account unique identifier")
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., description="Account username")
    full_name: str = Field(..., description="User full name")
    is_active: bool = Field(default=True, description="Account active status")
    is_verified: bool = Field(default=False, description="Email verification status")
    role: str = Field(default="user", description="User role (user, admin, developer)")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    github_username: Optional[str] = Field(None, description="GitHub username")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "id": "acc_123456789",
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "is_active": True,
                "is_verified": True,
                "role": "developer",
                "github_username": "johndoe",
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            }
        }


class AccountCreate(BaseModel):
    """Schema for creating new account."""
    
    email: EmailStr
    username: str
    full_name: str
    password: str = Field(..., min_length=8, description="Account password")
    github_username: Optional[str] = None
    

class AccountUpdate(BaseModel):
    """Schema for updating account."""
    
    full_name: Optional[str] = None
    github_username: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    

class AccountResponse(BaseModel):
    """Schema for account response (excludes sensitive data)."""
    
    id: str
    email: EmailStr
    username: str
    full_name: str
    is_active: bool
    is_verified: bool
    role: str
    github_username: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]
    preferences: Dict[str, Any]