from fastapi import HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
from config import API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE
import os
import json
import base64

# Function to validate API key (moved from config.py)
def validate_api_key(api_key_to_validate: str) -> bool:
    """
    Validate the provided API key against the configured key.
    """
    if not API_KEY: # API_KEY is imported from config
        # If no API key is configured, authentication is disabled (or treat as invalid)
        # Depending on desired behavior, for now, let's assume if API_KEY is not set, all keys are invalid unless it's an empty string match
        return False # Or True if you want to disable auth when API_KEY is not set
    return api_key_to_validate == API_KEY

# API Key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Dependency for API key validation
async def get_api_key(
    authorization: Optional[str] = Header(None),
    x_ip_token: Optional[str] = Header(None, alias="x-ip-token")
):
    # Check if Hugging Face auth is enabled
    if HUGGINGFACE:  # Use HUGGINGFACE from config
        if x_ip_token is None:
            raise HTTPException(
                status_code=401, # Unauthorised - because x-ip-token is missing
                detail="Missing x-ip-token header. This header is required for Hugging Face authentication."
            )

        try:
            # Decode JWT payload
            parts = x_ip_token.split('.')
            if len(parts) < 2:
                raise ValueError("Invalid JWT format: Not enough parts to extract payload.")
            payload_encoded = parts[1]
            # Add padding if necessary, as Python's base64.urlsafe_b64decode requires it
            payload_encoded += '=' * (-len(payload_encoded) % 4)
            decoded_payload_bytes = base64.urlsafe_b64decode(payload_encoded)
            payload = json.loads(decoded_payload_bytes.decode('utf-8'))
        except ValueError as ve:
            # Log server-side for debugging, but return a generic client error
            print(f"❌ [鉴权失败] 解析 Hugging Face x-ip-token 时格式无效：{ve}")
            raise HTTPException(status_code=400, detail=f"Invalid JWT format in x-ip-token: {str(ve)}")
        except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as e:
            print(f"❌ [鉴权失败] 解码 Hugging Face x-ip-token 载荷失败：{e}")
            raise HTTPException(status_code=400, detail=f"Malformed x-ip-token payload: {str(e)}")
        except Exception as e: # Catch any other unexpected errors during token processing
            print(f"❌ [鉴权异常] 处理 Hugging Face x-ip-token 时发生未预期异常：{e}")
            raise HTTPException(status_code=500, detail="Internal error processing x-ip-token.")

        error_in_token = payload.get("error")

        if error_in_token == "InvalidAccessToken":
            raise HTTPException(
                status_code=403,
                detail="Access denied: x-ip-token indicates 'InvalidAccessToken'."
            )
        elif error_in_token is None:  # JSON 'null' is Python's None
            # If error is null, auth is successful. Now check if HUGGINGFACE_API_KEY is configured.
            print("✅ [鉴权成功] Hugging Face x-ip-token 校验通过。")
            return HUGGINGFACE_API_KEY # Return the configured HUGGINGFACE_API_KEY
        else:
            # Any other non-null, non-"InvalidAccessToken" value in 'error' field
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: x-ip-token indicates an unhandled error: '{error_in_token}'."
            )
    else:
        # Fallback to Bearer token authentication if HUGGINGFACE env var is not "true"
        if authorization is None:
            detail_message = "Missing API key. Please include 'Authorization: Bearer YOUR_API_KEY' header."
            # Optionally, provide a hint if the HUGGINGFACE env var exists but is not "true"
            if os.getenv("HUGGINGFACE") is not None: # Check for existence, not value
                 detail_message += " (Note: HUGGINGFACE mode with x-ip-token is not currently active)."
            raise HTTPException(
                status_code=401,
                detail=detail_message
            )
        
        # Check if the header starts with "Bearer "
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key format. Use 'Authorization: Bearer YOUR_API_KEY'"
            )
        
        # Extract the API key
        api_key = authorization.replace("Bearer ", "")
        
        # Validate the API key
        if not validate_api_key(api_key): # Call local validate_api_key
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key