"""
Security Manager - RAG-The-Game-Changer

Production-grade security manager for authentication, authorization, and encryption.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    API_KEY = "api_key"
    JWT = "jwt"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class APIKey:
    """Represents an API key."""

    key_id: str
    key_value: str
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True
    rate_limit: int = 1000
    used_count: int = 0
    last_used: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    """Represents a user role with permissions."""

    name: str
    permissions: Set[str] = field(default_factory=set)
    resource_access: Set[str] = field(default_factory=set)


class SecurityManager:
    """Security manager for authentication, authorization, and encryption."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_keys: Dict[str, APIKey] = {}
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, str] = {}
        self._encryption_key = self.config.get("encryption_key", secrets.token_hex(32))
        self._jwt_secret = self.config.get("jwt_secret", secrets.token_urlsafe(64))
        self._rate_limits: Dict[str, List[float]] = {}
        self._audit_log: List[Dict[str, Any]] = []

        self._initialize_default_roles()

    def _initialize_default_roles(self):
        """Initialize default roles."""
        self.roles["admin"] = Role(
            name="admin", permissions={p.value for p in Permission}, resource_access={"*"}
        )
        self.roles["user"] = Role(
            name="user",
            permissions={Permission.READ.value, Permission.WRITE.value},
            resource_access={"documents", "queries"},
        )
        self.roles["guest"] = Role(
            name="guest", permissions={Permission.READ.value}, resource_access={"queries"}
        )

    async def create_api_key(
        self,
        user_id: str,
        role: str = "user",
        expires_in: Optional[int] = None,
        rate_limit: int = 1000,
    ) -> str:
        """Create a new API key."""
        key_id = f"key_{secrets.token_hex(8)}"
        key_value = f"rag_{secrets.token_urlsafe(32)}"

        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in

        api_key = APIKey(
            key_id=key_id,
            key_value=key_value,
            created_at=time.time(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            metadata={"user_id": user_id, "role": role},
        )

        self.api_keys[key_value] = api_key
        self.user_roles[key_id] = role

        self.audit_log("api_key_created", {"key_id": key_id, "user_id": user_id})

        logger.info(f"Created API key {key_id} for user {user_id}")
        return key_value

    async def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        try:
            if api_key not in self.api_keys:
                self.audit_log(
                    "auth_failed", {"reason": "invalid_key", "api_key_hash": self._hash(api_key)}
                )
                return None

            key = self.api_keys[api_key]

            # Check if key is active
            if not key.is_active:
                self.audit_log("auth_failed", {"reason": "key_inactive", "key_id": key.key_id})
                return None

            # Check if key is expired
            if key.expires_at and key.expires_at < time.time():
                self.audit_log("auth_failed", {"reason": "key_expired", "key_id": key.key_id})
                return None

            # Check rate limit
            if not self._check_rate_limit(api_key):
                self.audit_log(
                    "auth_failed", {"reason": "rate_limit_exceeded", "key_id": key.key_id}
                )
                return None

            # Update usage
            key.used_count += 1
            key.last_used = time.time()

            role_name = self.user_roles.get(key.key_id, "guest")
            role = self.roles.get(role_name, self.roles["guest"])

            self.audit_log(
                "auth_success", {"key_id": key.key_id, "user_id": key.metadata.get("user_id")}
            )

            return {
                "key_id": key.key_id,
                "user_id": key.metadata.get("user_id"),
                "role": role_name,
                "permissions": list(role.permissions),
                "resource_access": list(role.resource_access),
            }

        except Exception as e:
            logger.error(f"Error authenticating API key: {e}")
            return None

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check if API key has exceeded rate limit."""
        key = self.api_keys.get(api_key)
        if not key:
            return False

        now = time.time()
        window = 60  # 1 minute window

        if api_key not in self._rate_limits:
            self._rate_limits[api_key] = []

        # Clean old entries
        self._rate_limits[api_key] = [t for t in self._rate_limits[api_key] if now - t < window]

        # Check if under limit
        if len(self._rate_limits[api_key]) >= key.rate_limit:
            return False

        self._rate_limits[api_key].append(now)
        return True

    async def check_permission(self, user_info: Dict[str, Any], resource: str, action: str) -> bool:
        """Check if user has permission for resource and action."""
        try:
            role_name = user_info.get("role", "guest")
            role = self.roles.get(role_name)

            if not role:
                return False

            # Check permission
            if action not in role.permissions:
                return False

            # Check resource access
            if "*" not in role.resource_access and resource not in role.resource_access:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False

    def encrypt_data(self, data: str) -> str:
        """Encrypt data using AES."""
        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import pad
            import base64

            key = self._encryption_key[:32].encode()
            iv = secrets.token_bytes(16)

            cipher = AES.new(key, AES.MODE_CBC, iv)
            padded_data = pad(data.encode(), AES.block_size)
            encrypted = cipher.encrypt(padded_data)

            return base64.b64encode(iv + encrypted).decode()

        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using AES."""
        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import unpad
            import base64

            key = self._encryption_key[:32].encode()
            raw = base64.b64decode(encrypted_data.encode())

            iv = raw[:16]
            encrypted = raw[16:]

            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)

            return decrypted.decode()

        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

    def generate_jwt(self, user_id: str, expiry_hours: int = 24) -> str:
        """Generate JWT token for user."""
        try:
            import jwt

            payload = {
                "user_id": user_id,
                "exp": time.time() + (expiry_hours * 3600),
                "iat": time.time(),
            }

            token = jwt.encode(payload, self._jwt_secret, algorithm="HS256")

            self.audit_log("jwt_generated", {"user_id": user_id})

            return token

        except Exception as e:
            logger.error(f"Error generating JWT: {e}")
            raise

    def verify_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            import jwt

            payload = jwt.decode(token, self._jwt_secret, algorithms=["HS256"])

            self.audit_log("jwt_verified", {"user_id": payload.get("user_id")})

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        try:
            if api_key in self.api_keys:
                key_id = self.api_keys[api_key].key_id
                del self.api_keys[api_key]

                if key_id in self.user_roles:
                    del self.user_roles[key_id]

                self.audit_log("api_key_revoked", {"key_id": key_id})
                logger.info(f"Revoked API key {key_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False

    def audit_log(self, event: str, details: Optional[Dict[str, Any]] = None):
        """Log security event to audit log."""
        entry = {"timestamp": time.time(), "event": event, "details": details or {}}
        self._audit_log.append(entry)

        # Keep only last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_audit_log(
        self, event_filter: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        logs = self._audit_log

        if event_filter:
            logs = [log for log in logs if log.get("event") == event_filter]

        return logs[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get security manager statistics."""
        active_keys = sum(1 for k in self.api_keys.values() if k.is_active)

        return {
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_keys,
            "total_roles": len(self.roles),
            "total_audit_entries": len(self._audit_log),
            "users_with_roles": len(self.user_roles),
        }

    def _hash(self, data: str) -> str:
        """Hash data for logging."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
