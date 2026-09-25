from fastapi import Header, HTTPException, Depends
from typing import Optional

# Simple session token dictionary for local development
SESSION_STORE = {}

def get_current_student(x_student_id: Optional[str] = Header(None)) -> str:
    """Extracts student_id from request headers or enforces fallback."""
    if not x_student_id:
        # Fallback to default demo student if header is absent
        return "STU_DEMO_1001"
    return x_student_id

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple API Key gatekeeper for backend security."""
    valid_key = "counsellor-secret-token"
    if x_api_key and x_api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API Key credentials")
    return True