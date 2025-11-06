"""
Configuration Encryption System
Encrypts and decrypts sensitive configuration data
"""

import os
import base64
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import hashlib
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


class ConfigEncryption:
    """
    Configuration Encryption Manager
    
    Provides encryption/decryption for sensitive configuration data.
    Supports:
    - Fernet symmetric encryption with key derivation
    - AES-256-CBC encryption with PBKDF2 key derivation
    - Base64 encoding/decoding
    - Key management and rotation
    - Secure password-based encryption
    """
    
    def __init__(self, key_file: str = None):
        self.logger = logging.getLogger(__name__)
        self.key_file = Path(key_file) if key_file else Path("./config/.encryption_key")
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.master_key = None
        self.encryption_method = "fernet"  # Default method
        
        # Initialize encryption key
        self._initialize_key()
    
    def encrypt(self, data: str, password: str = None, method: str = None) -> str:
        """
        Encrypt sensitive configuration data
        
        Args:
            data: Data to encrypt
            password: Password for encryption (if None, uses master key)
            method: Encryption method ('fernet', 'aes256')
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            method = method or self.encryption_method
            
            if password:
                # Use password-based encryption
                if method == "fernet":
                    return self._encrypt_with_password_fernet(data, password)
                elif method == "aes256":
                    return self._encrypt_with_password_aes256(data, password)
                else:
                    raise ValueError(f"Unknown encryption method: {method}")
            else:
                # Use master key
                if method == "fernet":
                    return self._encrypt_with_master_fernet(data)
                elif method == "aes256":
                    return self._encrypt_with_master_aes256(data)
                else:
                    raise ValueError(f"Unknown encryption method: {method}")
                    
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str, password: str = None, method: str = None) -> str:
        """
        Decrypt sensitive configuration data
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            password: Password for decryption (if None, uses master key)
            method: Decryption method ('fernet', 'aes256')
            
        Returns:
            Decrypted string
        """
        try:
            method = method or self.encryption_method
            
            if password:
                # Use password-based decryption
                if method == "fernet":
                    return self._decrypt_with_password_fernet(encrypted_data, password)
                elif method == "aes256":
                    return self._decrypt_with_password_aes256(encrypted_data, password)
                else:
                    raise ValueError(f"Unknown decryption method: {method}")
            else:
                # Use master key
                if method == "fernet":
                    return self._decrypt_with_master_fernet(encrypted_data)
                elif method == "aes256":
                    return self._decrypt_with_master_aes256(encrypted_data)
                else:
                    raise ValueError(f"Unknown decryption method: {method}")
                    
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_config_section(self, config_data: Dict[str, Any], 
                              sensitive_keys: List[str] = None) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in configuration data
        
        Args:
            config_data: Configuration dictionary
            sensitive_keys: List of keys to encrypt (if None, auto-detect)
            
        Returns:
            Configuration with encrypted sensitive fields
        """
        if sensitive_keys is None:
            sensitive_keys = self._detect_sensitive_keys(config_data)
        
        encrypted_data = config_data.copy()
        
        def encrypt_recursive(obj, keys_to_check):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in keys_to_check:
                        if isinstance(value, str):
                            encrypted_value = self.encrypt(value)
                            obj[key] = f"encrypted:{encrypted_value}"
                    elif isinstance(value, (dict, list)):
                        encrypt_recursive(value, keys_to_check)
            elif isinstance(obj, list):
                for item in obj:
                    encrypt_recursive(item, keys_to_check)
        
        encrypt_recursive(encrypted_data, sensitive_keys)
        return encrypted_data
    
    def decrypt_config_section(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in configuration data
        
        Args:
            config_data: Configuration with encrypted fields
            
        Returns:
            Configuration with decrypted sensitive fields
        """
        decrypted_data = config_data.copy()
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith("encrypted:"):
                        try:
                            encrypted_part = value[10:]  # Remove "encrypted:" prefix
                            decrypted_value = self.decrypt(encrypted_part)
                            obj[key] = decrypted_value
                        except Exception as e:
                            self.logger.warning(f"Failed to decrypt field {key}: {e}")
                    elif isinstance(value, (dict, list)):
                        decrypt_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    decrypt_recursive(item)
        
        decrypt_recursive(decrypted_data)
        return decrypted_data
    
    def generate_new_key(self, method: str = "fernet") -> bytes:
        """Generate new encryption key"""
        if method == "fernet":
            return Fernet.generate_key()
        elif method == "aes256":
            return os.urandom(32)  # 256 bits
        else:
            raise ValueError(f"Unknown key generation method: {method}")
    
    def set_master_key(self, key: bytes):
        """Set master encryption key"""
        if not isinstance(key, bytes):
            raise ValueError("Key must be bytes")
        
        self.master_key = key
        self._save_key()
    
    def rotate_key(self, method: str = None) -> bool:
        """Rotate encryption key"""
        try:
            new_method = method or self.encryption_method
            new_key = self.generate_new_key(new_method)
            
            # Update master key
            self.master_key = new_key
            
            # Save new key
            self._save_key()
            
            self.logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return False
    
    def is_encrypted(self, value: str) -> bool:
        """Check if value is encrypted"""
        return isinstance(value, str) and value.startswith("encrypted:")
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get encryption system information"""
        return {
            "encryption_method": self.encryption_method,
            "key_file": str(self.key_file),
            "has_master_key": self.master_key is not None,
            "key_available": self.key_file.exists()
        }
    
    def _initialize_key(self):
        """Initialize encryption key"""
        try:
            if self.key_file.exists():
                with open(self.key_file, 'rb') as f:
                    self.master_key = f.read()
            else:
                # Generate new key
                self.master_key = self.generate_new_key()
                self._save_key()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption key: {e}")
            # Fallback: generate temporary key
            self.master_key = self.generate_new_key()
    
    def _save_key(self):
        """Save encryption key to file"""
        try:
            # Set secure permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(self.key_file, 0o600)
            
            with open(self.key_file, 'wb') as f:
                f.write(self.master_key)
                
        except Exception as e:
            self.logger.error(f"Failed to save encryption key: {e}")
    
    def _encrypt_with_master_fernet(self, data: str) -> str:
        """Encrypt with master key using Fernet"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        fernet = Fernet(self.master_key)
        encrypted = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_with_master_fernet(self, encrypted_data: str) -> str:
        """Decrypt with master key using Fernet"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        fernet = Fernet(self.master_key)
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def _encrypt_with_password_fernet(self, data: str, password: str) -> str:
        """Encrypt with password using Fernet"""
        # Derive key from password
        password_bytes = password.encode()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        
        # Encrypt data
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        
        # Combine salt and encrypted data
        result = base64.b64encode(salt + encrypted).decode()
        return result
    
    def _decrypt_with_password_fernet(self, encrypted_data: str, password: str) -> str:
        """Decrypt with password using Fernet"""
        # Decode combined data
        combined = base64.b64decode(encrypted_data.encode())
        salt = combined[:16]
        encrypted = combined[16:]
        
        # Derive key from password
        password_bytes = password.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        
        # Decrypt data
        fernet = Fernet(key)
        decrypted = fernet.decrypt(encrypted)
        return decrypted.decode()
    
    def _encrypt_with_master_aes256(self, data: str) -> str:
        """Encrypt with master key using AES-256-CBC"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Use master key directly for AES-256
        key = self.master_key[:32] if len(self.master_key) >= 32 else self.master_key.ljust(32, b'0')
        iv = os.urandom(16)
        
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data.encode()) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and encrypted data
        result = base64.b64encode(iv + encrypted).decode()
        return result
    
    def _decrypt_with_master_aes256(self, encrypted_data: str) -> str:
        """Decrypt with master key using AES-256-CBC"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Decode combined data
        combined = base64.b64decode(encrypted_data.encode())
        iv = combined[:16]
        encrypted = combined[16:]
        
        # Use master key directly for AES-256
        key = self.master_key[:32] if len(self.master_key) >= 32 else self.master_key.ljust(32, b'0')
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        
        # Unpad data
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        return data.decode()
    
    def _encrypt_with_password_aes256(self, data: str, password: str) -> str:
        """Encrypt with password using AES-256-CBC"""
        # Derive key from password
        password_bytes = password.encode()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password_bytes)
        
        iv = os.urandom(16)
        
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data.encode()) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine salt, IV, and encrypted data
        result = base64.b64encode(salt + iv + encrypted).decode()
        return result
    
    def _decrypt_with_password_aes256(self, encrypted_data: str, password: str) -> str:
        """Decrypt with password using AES-256-CBC"""
        # Decode combined data
        combined = base64.b64decode(encrypted_data.encode())
        salt = combined[:16]
        iv = combined[16:32]
        encrypted = combined[32:]
        
        # Derive key from password
        password_bytes = password.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password_bytes)
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        
        # Unpad data
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        return data.decode()
    
    def _detect_sensitive_keys(self, config_data: Dict[str, Any]) -> List[str]:
        """Auto-detect sensitive configuration keys"""
        sensitive_patterns = [
            'api_key', 'secret', 'password', 'token', 'key', 'credential',
            'private', 'secret_key', 'api_secret', 'auth_token', 'access_token',
            'refresh_token', 'bearer', 'jwt', 'oauth', 'client_secret'
        ]
        
        sensitive_keys = set()
        
        def check_keys(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = key.lower()
                    if any(pattern in key_lower for pattern in sensitive_patterns):
                        sensitive_keys.add(path + "." + key if path else key)
                    
                    # Recursively check nested objects
                    if isinstance(value, (dict, list)):
                        new_path = path + "." + key if path else key
                        check_keys(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        new_path = path + f"[{i}]"
                        check_keys(item, new_path)
        
        check_keys(config_data)
        return list(sensitive_keys)
    
    def encrypt_file(self, file_path: str, output_path: str = None, 
                    password: str = None) -> bool:
        """Encrypt configuration file"""
        try:
            if output_path is None:
                output_path = file_path + ".encrypted"
            
            # Read file
            with open(file_path, 'r') as f:
                data = f.read()
            
            # Encrypt
            encrypted_data = self.encrypt(data, password)
            
            # Write encrypted file
            with open(output_path, 'w') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {file_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            return False
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None, 
                    password: str = None) -> bool:
        """Decrypt configuration file"""
        try:
            if output_path is None:
                output_path = encrypted_file_path.replace('.encrypted', '')
            
            # Read encrypted file
            with open(encrypted_file_path, 'r') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = self.decrypt(encrypted_data, password)
            
            # Write decrypted file
            with open(output_path, 'w') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            return False
    
    def secure_delete_key_file(self) -> bool:
        """Securely delete encryption key file"""
        try:
            if not self.key_file.exists():
                return True
            
            # Overwrite file with random data multiple times
            file_size = self.key_file.stat().st_size
            
            with open(self.key_file, 'r+b') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete file
            self.key_file.unlink()
            
            # Reset master key
            self.master_key = None
            
            self.logger.info("Encryption key securely deleted")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to securely delete key file: {e}")
            return False


# Utility functions for configuration encryption
def create_secure_config(config_data: Dict[str, Any], 
                        sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """Create secure configuration with encrypted sensitive data"""
    encryption = ConfigEncryption()
    return encryption.encrypt_config_section(config_data, sensitive_keys)


def extract_secure_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sensitive data from secure configuration"""
    encryption = ConfigEncryption()
    return encryption.decrypt_config_section(config_data)


def is_config_secure(config_data: Dict[str, Any]) -> bool:
    """Check if configuration contains encrypted sensitive data"""
    encryption = ConfigEncryption()
    
    def has_encrypted(obj):
        if isinstance(obj, dict):
            return any(
                (isinstance(v, str) and encryption.is_encrypted(v)) or 
                has_encrypted(v) 
                for v in obj.values()
            )
        elif isinstance(obj, list):
            return any(has_encrypted(item) for item in obj)
        return False
    
    return has_encrypted(config_data)