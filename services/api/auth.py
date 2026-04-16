import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from services.api.security import normalize_identifier

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    # Dev fallback: rotates per process start and avoids committing fixed secrets.
    SECRET_KEY = secrets.token_urlsafe(32)

APP_ENV = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower()
if APP_ENV in {"prod", "production"} and not os.getenv("JWT_SECRET_KEY"):
    raise RuntimeError("JWT_SECRET_KEY must be set in production environments")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
ENABLE_DEMO_USERS = os.getenv(
    "ENABLE_DEMO_USERS",
    "false" if APP_ENV in {"prod", "production"} else "true",
).lower() == "true"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    roles: List[str] = Field(default_factory=list)


class User(BaseModel):
    username: str
    email: str
    roles: List[str] = Field(default_factory=lambda: ["viewer"])
    active: bool = True


class UserInDB(User):
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _decode_jwt_token(token: str) -> TokenData:
    """Decode and validate a JWT token string, returning TokenData."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", [])

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenData(username=username, roles=roles)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    return _decode_jwt_token(credentials.credentials)


def get_current_user_or_token_param(
    request: "Request",
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
) -> TokenData:
    """Authenticate via Authorization header or ?token= query parameter.

    This is intended for SSE endpoints where the browser's native EventSource
    API cannot send custom headers.  The Authorization header is preferred;
    the query-param is a fallback.
    """
    if credentials is not None:
        return _decode_jwt_token(credentials.credentials)

    token = request.query_params.get("token")
    if token:
        return _decode_jwt_token(token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (Authorization header or ?token= query parameter)",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(required_role: str):
    async def check_role(current_user: TokenData = Depends(get_current_user)):
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required role: {required_role}"
            )
        return current_user
    
    return check_role


def _build_demo_users() -> dict:
    users = {}
    specs = [
        ("admin", "admin@threat-detector.local", ["admin", "analyst", "viewer"], "DEMO_ADMIN_PASSWORD"),
        ("analyst", "analyst@threat-detector.local", ["analyst", "viewer"], "DEMO_ANALYST_PASSWORD"),
        ("viewer", "viewer@threat-detector.local", ["viewer"], "DEMO_VIEWER_PASSWORD"),
    ]

    for username, email, roles, env_var in specs:
        password = os.getenv(env_var)
        if not password:
            continue
        users[username] = {
            "username": username,
            "email": email,
            "hashed_password": get_password_hash(password),
            "roles": roles,
            "active": True,
        }

    return users


# Demo users are optional and only enabled when explicitly configured.
DEMO_USERS = _build_demo_users()


def get_user_from_db(username: str) -> Optional[UserInDB]:
    username = normalize_identifier(username, field_name="username", max_length=64)
    if not ENABLE_DEMO_USERS:
        return None
    if username in DEMO_USERS:
        user_dict = DEMO_USERS[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    username = normalize_identifier(username, field_name="username", max_length=64)
    user = get_user_from_db(username)
    if not user:
        return None
    if not user.active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
