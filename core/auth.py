"""
Enterprise Authentication & Cryptographic Security
Implements OAuth 2.1 PKCE (Proof Key for Code Exchange), SHA-256 Code Challenges,
ephemeral JWT session issuance, and granular Scope/Role-Based Access Control (RBAC).
"""

import base64
import hashlib
import os
import time
from typing import Dict, List, Optional
import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from config.settings import settings

security_bearer = HTTPBearer()


# ==============================================================================
# 1. OAuth 2.1 PKCE Security Utilities
# ==============================================================================

def generate_pkce_verifier(length: int = 64) -> str:
    """
    Generates a cryptographically secure high-entropy random string for PKCE.
    Length must be between 43 and 128 characters according to RFC 7636.
    """
    raw_bytes = os.urandom(length)
    return base64.urlsafe_b64encode(raw_bytes).decode("utf-8").rstrip("=")[:length]


def derive_code_challenge(verifier: str) -> str:
    """
    Derives the SHA-256 code challenge from a code verifier:
    BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def verify_pkce(verifier: str, challenge: str) -> bool:
    """
    Validates if the client's code verifier matches the initial code challenge.
    """
    expected_challenge = derive_code_challenge(verifier)
    return expected_challenge == challenge


# ==============================================================================
# 2. JWT Session Token Issuance & Verification
# ==============================================================================

class TokenPayload(BaseModel):
    """Decoded JWT claim payload."""
    sub: str = Field(..., description="User ID subject.")
    tenant_id: str = Field(default="default_tenant", description="Tenant / Organization ID.")
    scopes: List[str] = Field(default_factory=lambda: ["read", "order", "checkout"], description="Permissions granted.")
    exp: int = Field(..., description="Expiration epoch timestamp.")


def create_access_token(user_id: str, tenant_id: str = "default_tenant", scopes: Optional[List[str]] = None, expires_in: int = 3600) -> str:
    """
    Generates a cryptographically signed JWT token with expiry.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "scopes": scopes if scopes is not None else ["read", "order", "checkout"],
        "iat": now,
        "exp": now + expires_in
    }
    token = jwt.encode(payload, settings.APP_SECRET_KEY, algorithm="HS256")
    return token


def decode_access_token(token: str) -> TokenPayload:
    """
    Decodes and validates the signature and expiration of a JWT access token.
    Raises HTTPException(401) on failure.
    """
    try:
        data = jwt.decode(token, settings.APP_SECRET_KEY, algorithms=["HS256"])
        return TokenPayload(**data)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please re-authenticate."
        )
    except (jwt.InvalidTokenError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authorization token: {str(exc)}"
        )


async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Security(security_bearer)) -> str:
    """
    FastAPI dependency that extracts and validates the Bearer JWT token from HTTP headers.
    Returns the authenticated user_id (Authentication).
    """
    token = credentials.credentials
    payload = decode_access_token(token)
    return payload.sub


def require_scopes(required_scopes: List[str]):
    """
    FastAPI dependency factory enforcing granular RBAC/Scope Authorization.
    Verifies that the decoded JWT contains all necessary permissions for the endpoint.
    Raises HTTPException(403 Forbidden) if any required scope is missing.
    """
    async def scope_checker(credentials: HTTPAuthorizationCredentials = Security(security_bearer)) -> TokenPayload:
        token = credentials.credentials
        payload = decode_access_token(token)
        
        user_scopes = set(payload.scopes)
        missing_scopes = [s for s in required_scopes if s not in user_scopes]
        
        if missing_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Token missing required permissions/scopes: {missing_scopes}"
            )
        return payload

    return scope_checker
