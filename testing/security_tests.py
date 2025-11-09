"""
Security Test Suite
Penetration testing, vulnerability scanning, and security validation.
"""

import unittest
import asyncio
import logging
import json
import time
import hashlib
import secrets
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import ssl
import socket
from datetime import datetime
import base64
import urllib.parse
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityTestCategory(Enum):
    """Security test categories."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    SESSION_MANAGEMENT = "session_management"
    CRYPTOGRAPHY = "cryptography"
    NETWORK_SECURITY = "network_security"
    DATA_PROTECTION = "data_protection"
    VULNERABILITY_SCANNING = "vulnerability_scanning"


@dataclass
class SecurityFinding:
    """Security finding container."""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    category: SecurityTestCategory
    affected_url: Optional[str] = None
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    timestamp: datetime = None


@dataclass
class SecurityTestResult:
    """Security test result container."""
    test_name: str
    category: SecurityTestCategory
    passed: bool
    findings: List[SecurityFinding]
    duration: float
    timestamp: datetime = None


class SecurityTestSuite:
    """Comprehensive security testing suite."""
    
    def __init__(self, target_url: str = "http://localhost:8000"):
        self.target_url = target_url
        self.base_url = target_url.rstrip('/')
        self.findings: List[SecurityFinding] = []
        self.test_results: List[SecurityTestResult] = []
        self.scan_results: Dict[str, Any] = {}
        
        # Security test payloads
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "admin'--",
            "' UNION SELECT NULL--",
            "'; DROP TABLE users;--"
        ]
        
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`",
            "$(whoami)"
        ]
        
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        # Known vulnerability patterns
        self.vulnerability_patterns = {
            "sql_error": [
                r"SQL syntax.*MySQL",
                r"PostgreSQL.*ERROR",
                r"Warning.*\Wmysqli?_\w+\(",
                r"valid MySQL result",
                r"MySqlClient\.",""
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=\s*[\"']?[^\"'>]*[\"']?",
                r"<img[^>]*src\s*=\s*[\"']javascript:",
            ],
            "path_traversal": [
                r"\.\./\.\./\.\./",
                r"\.\.\\\.\.\\\.\.\\",
                r"%2e%2e%2f%2e%2e%2f",
                r"\.\.%2f\.\.%2f"
            ],
            "command_injection": [
                r"[;&|`$]\s*(ls|whoami|cat|id)",
                r"\|\|.*(ls|whoami|cat|id)",
                r"&&.*(ls|whoami|cat|id)"
            ],
            "ssl_issues": [
                r"Self-signed certificate",
                r"certificate verify failed",
                r"SSL: CERTIFICATE_VERIFY_FAILED"
            ]
        }
    
    async def run_comprehensive_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scanning."""
        logger.info("Starting comprehensive security scan...")
        
        overall_start_time = time.time()
        results = []
        
        # Authentication tests
        auth_results = await self.run_authentication_tests()
        results.extend(auth_results)
        
        # Authorization tests
        authz_results = await self.run_authorization_tests()
        results.extend(authz_results)
        
        # Input validation tests
        input_results = await self.run_input_validation_tests()
        results.extend(input_results)
        
        # Session management tests
        session_results = await self.run_session_management_tests()
        results.extend(session_results)
        
        # Network security tests
        network_results = await self.run_network_security_tests()
        results.extend(network_results)
        
        # Vulnerability scanning
        vuln_results = await self.run_vulnerability_scanning()
        results.extend(vuln_results)
        
        overall_duration = time.time() - overall_start_time
        
        # Generate comprehensive security report
        report = {
            "timestamp": datetime.now().isoformat(),
            "target_url": self.target_url,
            "scan_duration": overall_duration,
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.passed]),
            "failed_tests": len([r for r in results if not r.passed]),
            "total_findings": len(self.findings),
            "findings_by_severity": self._count_findings_by_severity(),
            "findings_by_category": self._count_findings_by_category(),
            "test_results": [self._result_to_dict(r) for r in results],
            "security_score": self._calculate_security_score(results),
            "recommendations": self._generate_security_recommendations()
        }
        
        logger.info(f"Security scan completed in {overall_duration:.2f} seconds")
        return report
    
    async def run_authentication_tests(self) -> List[SecurityTestResult]:
        """Run authentication security tests."""
        logger.info("Running authentication tests...")
        results = []
        
        # Test 1: Default credentials
        result = await self.test_default_credentials()
        results.append(result)
        
        # Test 2: Weak password policies
        result = await self.test_password_policies()
        results.append(result)
        
        # Test 3: Brute force protection
        result = await self.test_brute_force_protection()
        results.append(result)
        
        # Test 4: Authentication bypass
        result = await self.test_authentication_bypass()
        results.append(result)
        
        return results
    
    async def test_default_credentials(self) -> SecurityTestResult:
        """Test for default credentials."""
        start_time = time.time()
        findings = []
        
        default_credentials = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("root", "root"),
            ("user", "user"),
            ("guest", "guest"),
            ("test", "test")
        ]
        
        for username, password in default_credentials:
            try:
                async with aiohttp.ClientSession() as session:
                    # Try common login endpoints
                    login_endpoints = [
                        "/login",
                        "/admin/login",
                        "/auth/login",
                        "/signin"
                    ]
                    
                    for endpoint in login_endpoints:
                        url = f"{self.base_url}{endpoint}"
                        
                        # Try form-based login
                        data = {"username": username, "password": password}
                        async with session.post(url, data=data) as response:
                            if response.status == 200:
                                text = await response.text()
                                # Check if login was successful
                                if "dashboard" in text.lower() or "welcome" in text.lower():
                                    finding = SecurityFinding(
                                        id=f"auth_default_{username}_{password}",
                                        title="Default Credentials Found",
                                        description=f"Default credentials {username}/{password} work at {endpoint}",
                                        severity=VulnerabilitySeverity.HIGH,
                                        category=SecurityTestCategory.AUTHENTICATION,
                                        affected_url=url,
                                        remediation="Remove default credentials and enforce strong password policies"
                                    )
                                    findings.append(finding)
                                    logger.warning(f"Default credentials found: {username}/{password} at {url}")
                                    break
                        
                        # Try JSON-based login
                        json_data = {"username": username, "password": password}
                        headers = {"Content-Type": "application/json"}
                        async with session.post(url, json=json_data, headers=headers) as response:
                            if response.status == 200:
                                text = await response.text()
                                try:
                                    json_response = json.loads(text)
                                    if "token" in json_response or "success" in str(json_response).lower():
                                        finding = SecurityFinding(
                                            id=f"auth_default_json_{username}_{password}",
                                            title="Default Credentials Found (JSON)",
                                            description=f"Default credentials {username}/{password} work via JSON API at {endpoint}",
                                            severity=VulnerabilitySeverity.HIGH,
                                            category=SecurityTestCategory.AUTHENTICATION,
                                            affected_url=url,
                                            remediation="Remove default credentials and enforce strong password policies"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"Default credentials found (JSON): {username}/{password} at {url}")
                                        break
                                except json.JSONDecodeError:
                                    pass
                
            except Exception as e:
                logger.debug(f"Error testing credentials {username}/{password}: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Default Credentials Test",
            category=SecurityTestCategory.AUTHENTICATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_password_policies(self) -> SecurityTestResult:
        """Test password policy enforcement."""
        start_time = time.time()
        findings = []
        
        # Try weak passwords
        weak_passwords = [
            "123456",
            "password",
            "12345678",
            "qwerty",
            "abc123",
            "111111",
            "password123",
            "admin",
            "123456789"
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try to find registration endpoint
                register_endpoints = ["/register", "/signup", "/auth/register"]
                
                for endpoint in register_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    for password in weak_passwords:
                        try:
                            # Try to register with weak password
                            data = {
                                "username": f"test_{secrets.token_hex(4)}",
                                "password": password,
                                "email": f"test_{secrets.token_hex(4)}@example.com"
                            }
                            
                            async with session.post(url, data=data) as response:
                                if response.status == 201 or response.status == 200:
                                    text = await response.text()
                                    
                                    # Check if weak password was accepted
                                    if "success" in text.lower() or "created" in text.lower():
                                        finding = SecurityFinding(
                                            id=f"weak_password_{password}",
                                            title="Weak Password Policy",
                                            description=f"Weak password '{password}' was accepted during registration",
                                            severity=VulnerabilitySeverity.MEDIUM,
                                            category=SecurityTestCategory.AUTHENTICATION,
                                            affected_url=url,
                                            remediation="Implement strong password policy (minimum 8 characters, mixed case, numbers, special characters)"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"Weak password accepted: {password}")
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error testing password {password}: {str(e)}")
                            continue
                    
                    if findings:
                        break  # Found issues with this endpoint
                        
        except Exception as e:
            logger.error(f"Error in password policy test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Password Policy Test",
            category=SecurityTestCategory.AUTHENTICATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_brute_force_protection(self) -> SecurityTestResult:
        """Test for brute force protection."""
        start_time = time.time()
        findings = []
        
        try:
            # Attempt multiple failed login attempts
            failed_attempts = 0
            max_attempts = 10
            
            async with aiohttp.ClientSession() as session:
                for i in range(max_attempts):
                    # Try random credentials
                    username = f"fake_user_{i}"
                    password = f"wrong_password_{i}"
                    
                    data = {"username": username, "password": password}
                    
                    async with session.post(f"{self.base_url}/login", data=data) as response:
                        if response.status == 401:
                            failed_attempts += 1
                        elif response.status == 429:  # Too Many Requests
                            # Brute force protection detected
                            logger.info("Brute force protection is working (429 status)")
                            break
                        elif response.status == 200:
                            # Check if login was successful (shouldn't be with fake creds)
                            text = await response.text()
                            if "success" in text.lower():
                                finding = SecurityFinding(
                                    id="brute_force_no_protection",
                                    title="No Brute Force Protection",
                                    description="No rate limiting or account lockout detected after multiple failed attempts",
                                    severity=VulnerabilitySeverity.HIGH,
                                    category=SecurityTestCategory.AUTHENTICATION,
                                    affected_url=f"{self.base_url}/login",
                                    remediation="Implement rate limiting, account lockout after failed attempts, and CAPTCHA"
                                )
                                findings.append(finding)
                                break
                        
                        # Small delay between attempts
                        await asyncio.sleep(0.5)
            
            if failed_attempts == max_attempts:
                finding = SecurityFinding(
                    id="brute_force_likely_no_protection",
                    title="Potential Lack of Brute Force Protection",
                    description=f"All {max_attempts} attempts received 401 responses, no rate limiting detected",
                    severity=VulnerabilitySeverity.MEDIUM,
                    category=SecurityTestCategory.AUTHENTICATION,
                    affected_url=f"{self.base_url}/login",
                    remediation="Implement rate limiting and account lockout mechanisms"
                )
                findings.append(finding)
                
        except Exception as e:
            logger.error(f"Error in brute force protection test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Brute Force Protection Test",
            category=SecurityTestCategory.AUTHENTICATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_authentication_bypass(self) -> SecurityTestResult:
        """Test for authentication bypass vulnerabilities."""
        start_time = time.time()
        findings = []
        
        # Test common authentication bypass techniques
        bypass_techniques = [
            # SQL injection bypass
            "admin'--",
            "admin' OR '1'='1",
            "' OR 1=1--",
            
            # JSON injection
            '{"username": "admin", "password": {"$ne": null}}',
            '{"username": "admin", "password": {"$ne": ""}}',
            
            # Special characters
            "admin\x00",
            "\x00admin",
            "admin\t",
            "admin\n"
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                login_endpoint = f"{self.base_url}/login"
                
                for technique in bypass_techniques:
                    # Test with JSON
                    json_data = {"username": technique, "password": "anything"}
                    headers = {"Content-Type": "application/json"}
                    
                    async with session.post(login_endpoint, json=json_data, headers=headers) as response:
                        if response.status == 200:
                            text = await response.text()
                            
                            # Check for successful authentication indicators
                            if any(indicator in text.lower() for indicator in ["success", "token", "welcome", "dashboard"]):
                                # This could be an authentication bypass
                                finding = SecurityFinding(
                                    id=f"auth_bypass_{hashlib.md5(technique.encode()).hexdigest()[:8]}",
                                    title="Potential Authentication Bypass",
                                    description=f"Authentication bypass technique '{technique}' may be working",
                                    severity=VulnerabilitySeverity.CRITICAL,
                                    category=SecurityTestCategory.AUTHENTICATION,
                                    affected_url=login_endpoint,
                                    remediation="Fix authentication logic to properly validate credentials"
                                )
                                findings.append(finding)
                                logger.warning(f"Potential auth bypass with: {technique}")
                                break
                            
        except Exception as e:
            logger.error(f"Error in authentication bypass test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Authentication Bypass Test",
            category=SecurityTestCategory.AUTHENTICATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def run_input_validation_tests(self) -> List[SecurityTestResult]:
        """Run input validation security tests."""
        logger.info("Running input validation tests...")
        results = []
        
        # SQL Injection tests
        sql_result = await self.test_sql_injection()
        results.append(sql_result)
        
        # XSS tests
        xss_result = await self.test_cross_site_scripting()
        results.append(xss_result)
        
        # Command injection tests
        cmd_result = await self.test_command_injection()
        results.append(cmd_result)
        
        # Path traversal tests
        path_result = await self.test_path_traversal()
        results.append(path_result)
        
        return results
    
    async def test_sql_injection(self) -> SecurityTestResult:
        """Test for SQL injection vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test various endpoints for SQL injection
                test_endpoints = [
                    "/login",
                    "/search",
                    "/user?id=1",
                    "/api/users/1",
                    "/products?category=1"
                ]
                
                for endpoint in test_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    for payload in self.sql_injection_payloads:
                        # Test via GET parameters
                        test_url = f"{url}?q={urllib.parse.quote(payload)}"
                        try:
                            async with session.get(test_url) as response:
                                text = await response.text()
                                
                                # Check for SQL error messages
                                for pattern in self.vulnerability_patterns["sql_error"]:
                                    if re.search(pattern, text, re.IGNORECASE):
                                        finding = SecurityFinding(
                                            id=f"sql_injection_{endpoint}",
                                            title="SQL Injection Vulnerability",
                                            description=f"SQL injection payload '{payload}' triggered database error",
                                            severity=VulnerabilitySeverity.CRITICAL,
                                            category=SecurityTestCategory.INPUT_VALIDATION,
                                            affected_url=test_url,
                                            remediation="Use parameterized queries and input sanitization"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"SQL injection vulnerability at {test_url}")
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error testing SQL injection at {test_url}: {str(e)}")
                        
                        # Test via POST parameters
                        if "/login" in endpoint:
                            data = {"username": payload, "password": "test"}
                            try:
                                async with session.post(url, data=data) as response:
                                    text = await response.text()
                                    
                                    for pattern in self.vulnerability_patterns["sql_error"]:
                                        if re.search(pattern, text, re.IGNORECASE):
                                            finding = SecurityFinding(
                                                id=f"sql_injection_post_{endpoint}",
                                                title="SQL Injection Vulnerability (POST)",
                                                description=f"SQL injection payload '{payload}' in POST data triggered database error",
                                                severity=VulnerabilitySeverity.CRITICAL,
                                                category=SecurityTestCategory.INPUT_VALIDATION,
                                                affected_url=url,
                                                remediation="Use parameterized queries and input sanitization"
                                            )
                                            findings.append(finding)
                                            logger.warning(f"SQL injection vulnerability (POST) at {url}")
                                            break
                                            
                            except Exception as e:
                                logger.debug(f"Error testing POST SQL injection at {url}: {str(e)}")
                        
                        if findings:
                            break  # Found vulnerabilities at this endpoint
                    
                    if findings:
                        break  # Stop testing other endpoints if we found vulnerabilities
                        
        except Exception as e:
            logger.error(f"Error in SQL injection test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="SQL Injection Test",
            category=SecurityTestCategory.INPUT_VALIDATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_cross_site_scripting(self) -> SecurityTestResult:
        """Test for XSS vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test endpoints that reflect user input
                test_endpoints = [
                    "/search?q=test",
                    "/profile?name=test",
                    "/comment?text=test"
                ]
                
                for endpoint in test_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    for payload in self.xss_payloads:
                        # Test via GET parameters
                        test_url = f"{url}?q={urllib.parse.quote(payload)}"
                        try:
                            async with session.get(test_url) as response:
                                text = await response.text()
                                
                                # Check if payload is reflected in response
                                if payload in text:
                                    # Check if it's properly encoded
                                    encoded_payload = urllib.parse.quote(payload, safe='')
                                    if encoded_payload not in text:
                                        finding = SecurityFinding(
                                            id=f"xss_{endpoint}",
                                            title="Cross-Site Scripting (XSS) Vulnerability",
                                            description=f"XSS payload '{payload}' is reflected without proper encoding",
                                            severity=VulnerabilitySeverity.HIGH,
                                            category=SecurityTestCategory.INPUT_VALIDATION,
                                            affected_url=test_url,
                                            remediation="Implement proper output encoding and Content Security Policy (CSP)"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"XSS vulnerability at {test_url}")
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error testing XSS at {test_url}: {str(e)}")
                        
                        if findings:
                            break  # Found XSS at this endpoint
                    
                    if findings:
                        break  # Stop testing other endpoints if we found XSS
                        
        except Exception as e:
            logger.error(f"Error in XSS test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Cross-Site Scripting Test",
            category=SecurityTestCategory.INPUT_VALIDATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_command_injection(self) -> SecurityTestResult:
        """Test for command injection vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test endpoints that might execute system commands
                test_endpoints = [
                    "/ping",
                    "/traceroute",
                    "/nslookup",
                    "/execute"
                ]
                
                for endpoint in test_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    for payload in self.command_injection_payloads:
                        # Test via GET parameters
                        test_url = f"{url}?host={urllib.parse.quote(payload)}"
                        try:
                            async with session.get(test_url, timeout=10) as response:
                                text = await response.text()
                                
                                # Check for command execution indicators
                                for pattern in self.vulnerability_patterns["command_injection"]:
                                    if re.search(pattern, text, re.IGNORECASE):
                                        finding = SecurityFinding(
                                            id=f"command_injection_{endpoint}",
                                            title="Command Injection Vulnerability",
                                            description=f"Command injection payload '{payload}' appears to be executed",
                                            severity=VulnerabilitySeverity.CRITICAL,
                                            category=SecurityTestCategory.INPUT_VALIDATION,
                                            affected_url=test_url,
                                            remediation="Validate and sanitize input, use parameterized commands"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"Command injection vulnerability at {test_url}")
                                        break
                                        
                        except asyncio.TimeoutError:
                            # Timeout might indicate command execution
                            logger.debug(f"Timeout testing command injection at {test_url}")
                        except Exception as e:
                            logger.debug(f"Error testing command injection at {test_url}: {str(e)}")
                        
                        if findings:
                            break  # Found command injection at this endpoint
                    
                    if findings:
                        break  # Stop testing other endpoints if we found vulnerabilities
                        
        except Exception as e:
            logger.error(f"Error in command injection test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Command Injection Test",
            category=SecurityTestCategory.INPUT_VALIDATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def test_path_traversal(self) -> SecurityTestResult:
        """Test for path traversal vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test endpoints that handle file paths
                test_endpoints = [
                    "/download?file=test.txt",
                    "/image?id=1",
                    "/document?path=test",
                    "/api/files/1"
                ]
                
                for endpoint in test_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    for payload in self.path_traversal_payloads:
                        # Test via GET parameters
                        test_url = f"{url}?file={urllib.parse.quote(payload)}"
                        try:
                            async with session.get(test_url) as response:
                                text = await response.text()
                                
                                # Check for path traversal indicators
                                for pattern in self.vulnerability_patterns["path_traversal"]:
                                    if re.search(pattern, text, re.IGNORECASE):
                                        finding = SecurityFinding(
                                            id=f"path_traversal_{endpoint}",
                                            title="Path Traversal Vulnerability",
                                            description=f"Path traversal payload '{payload}' allowed access to restricted files",
                                            severity=VulnerabilitySeverity.HIGH,
                                            category=SecurityTestCategory.INPUT_VALIDATION,
                                            affected_url=test_url,
                                            remediation="Validate file paths, use whitelist of allowed files"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"Path traversal vulnerability at {test_url}")
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error testing path traversal at {test_url}: {str(e)}")
                        
                        if findings:
                            break  # Found path traversal at this endpoint
                    
                    if findings:
                        break  # Stop testing other endpoints if we found vulnerabilities
                        
        except Exception as e:
            logger.error(f"Error in path traversal test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Path Traversal Test",
            category=SecurityTestCategory.INPUT_VALIDATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def run_authorization_tests(self) -> List[SecurityTestResult]:
        """Run authorization security tests."""
        logger.info("Running authorization tests...")
        results = []
        
        # IDOR (Insecure Direct Object Reference) test
        idor_result = await self.test_insecure_direct_object_references()
        results.append(idor_result)
        
        return results
    
    async def test_insecure_direct_object_references(self) -> SecurityTestResult:
        """Test for IDOR vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test endpoints that use object IDs
                test_endpoints = [
                    "/api/users/1",
                    "/api/orders/1",
                    "/api/documents/1",
                    "/profile/1",
                    "/dashboard/1"
                ]
                
                for endpoint in test_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    
                    # Test common IDOR techniques
                    idor_tests = [
                        "999999",
                        "-1",
                        "0",
                        "null",
                        "1' OR '1'='1",
                        "1; DROP TABLE users;--"
                    ]
                    
                    for test_id in idor_tests:
                        test_url = endpoint.replace("1", test_id)
                        full_url = f"{self.base_url}{test_url}"
                        
                        try:
                            async with session.get(full_url) as response:
                                # Check if we can access data we shouldn't
                                if response.status == 200:
                                    text = await response.text()
                                    
                                    # Look for data that shouldn't be accessible
                                    sensitive_indicators = [
                                        "password",
                                        "email",
                                        "ssn",
                                        "credit_card",
                                        "private",
                                        "confidential"
                                    ]
                                    
                                    if any(indicator in text.lower() for indicator in sensitive_indicators):
                                        finding = SecurityFinding(
                                            id=f"idor_{test_id}",
                                            title="Insecure Direct Object Reference (IDOR)",
                                            description=f"Potential IDOR vulnerability - able to access data with ID '{test_id}'",
                                            severity=VulnerabilitySeverity.HIGH,
                                            category=SecurityTestCategory.AUTHORIZATION,
                                            affected_url=full_url,
                                            remediation="Implement proper authorization checks for all object access"
                                        )
                                        findings.append(finding)
                                        logger.warning(f"Potential IDOR vulnerability at {full_url}")
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error testing IDOR at {full_url}: {str(e)}")
                        
                        if findings:
                            break  # Found IDOR at this endpoint
                    
                    if findings:
                        break  # Stop testing other endpoints if we found IDOR
                        
        except Exception as e:
            logger.error(f"Error in IDOR test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Insecure Direct Object References Test",
            category=SecurityTestCategory.AUTHORIZATION,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def run_session_management_tests(self) -> List[SecurityTestResult]:
        """Run session management security tests."""
        logger.info("Running session management tests...")
        results = []
        
        # Session fixation test
        fixation_result = await self.test_session_fixation()
        results.append(fixation_result)
        
        return result
    
    async def test_session_fixation(self) -> SecurityTestResult:
        """Test for session fixation vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            # Generate a test session ID
            test_session_id = secrets.token_hex(32)
            
            async with aiohttp.ClientSession() as session:
                # Step 1: Get initial page without session
                async with session.get(self.base_url) as response:
                    initial_cookies = dict(response.cookies)
                
                # Step 2: Try to set a specific session ID
                session.cookies.set("sessionid", test_session_id)
                
                # Step 3: Try to authenticate
                auth_data = {"username": "test", "password": "test"}
                async with session.post(f"{self.base_url}/login", data=auth_data) as response:
                    auth_cookies = dict(session.cookies)
                
                # Step 4: Check if our session ID is still active
                async with session.get(self.base_url) as response:
                    final_cookies = dict(response.cookies)
                
                # Check if session ID was changed after authentication
                if "sessionid" in auth_cookies and "sessionid" in final_cookies:
                    if auth_cookies["sessionid"] == final_cookies["sessionid"] == test_session_id:
                        finding = SecurityFinding(
                            id="session_fixation",
                            title="Session Fixation Vulnerability",
                            description="Session ID does not change after authentication, allowing session fixation attacks",
                            severity=VulnerabilitySeverity.MEDIUM,
                            category=SecurityTestCategory.SESSION_MANAGEMENT,
                            affected_url=self.base_url,
                            remediation="Regenerate session ID after authentication"
                        )
                        findings.append(finding)
                        logger.warning("Session fixation vulnerability detected")
                
        except Exception as e:
            logger.error(f"Error in session fixation test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Session Fixation Test",
            category=SecurityTestCategory.SESSION_MANAGEMENT,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def run_network_security_tests(self) -> List[SecurityTestResult]:
        """Run network security tests."""
        logger.info("Running network security tests...")
        results = []
        
        # SSL/TLS test
        ssl_result = await self.test_ssl_configuration()
        results.append(ssl_result)
        
        return results
    
    async def test_ssl_configuration(self) -> SecurityTestResult:
        """Test SSL/TLS configuration."""
        start_time = time.time()
        findings = []
        
        try:
            # Parse URL to get host and port
            parsed_url = urllib.parse.urlparse(self.target_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            # Test SSL certificate
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Check certificate validity
                    import datetime
                    now = datetime.datetime.now()
                    
                    if cert:
                        # Check expiration
                        not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        if not_after < now:
                            finding = SecurityFinding(
                                id="ssl_expired",
                                title="SSL Certificate Expired",
                                description="SSL certificate has expired",
                                severity=VulnerabilitySeverity.HIGH,
                                category=SecurityTestCategory.NETWORK_SECURITY,
                                remediation="Renew SSL certificate"
                            )
                            findings.append(finding)
                        
                        # Check if certificate matches hostname
                        cert_hostnames = cert.get('subjectAltName', [])
                        common_name = cert.get('subject', [{}])[0].get('commonName', '')
                        
                        if hostname not in [name[1] for name in cert_hostnames] and common_name != hostname:
                            finding = SecurityFinding(
                                id="ssl_hostname_mismatch",
                                title="SSL Certificate Hostname Mismatch",
                                description="SSL certificate does not match the requested hostname",
                                severity=VulnerabilitySeverity.HIGH,
                                category=SecurityTestCategory.NETWORK_SECURITY,
                                remediation="Obtain SSL certificate for the correct hostname"
                            )
                            findings.append(finding)
            
            # Test for weak SSL/TLS versions
            try:
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    try:
                        # Try SSLv2/SSLv3 (should fail)
                        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                        
                        with context.wrap_socket(sock, server_hostname=hostname, 
                                               ssl_version=ssl.PROTOCOL_TLS) as ssock:
                            # Connected successfully - this is good (uses modern TLS)
                            pass
                            
                    except ssl.SSLError:
                        # SSL/TLS error - could indicate weak protocols are disabled (good)
                        pass
                        
            except Exception as e:
                logger.debug(f"Error testing SSL versions: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in SSL configuration test: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="SSL Configuration Test",
            category=SecurityTestCategory.NETWORK_SECURITY,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def run_vulnerability_scanning(self) -> List[SecurityTestResult]:
        """Run vulnerability scanning."""
        logger.info("Running vulnerability scanning...")
        results = []
        
        result = await self.scan_common_vulnerabilities()
        results.append(result)
        
        return results
    
    async def scan_common_vulnerabilities(self) -> SecurityTestResult:
        """Scan for common web application vulnerabilities."""
        start_time = time.time()
        findings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check for common vulnerability endpoints
                vuln_endpoints = [
                    "/admin",
                    "/administrator",
                    "/wp-admin",
                    "/.git",
                    "/.env",
                    "/config.php",
                    "/backup.sql",
                    "/test.php",
                    "/info.php",
                    "/phpinfo.php"
                ]
                
                for endpoint in vuln_endpoints:
                    url = f"{self.base_url}{endpoint}"
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                text = await response.text()
                                
                                # Check for sensitive information leakage
                                if any(indicator in text.lower() for indicator in [
                                    "mysql_connect",
                                    "database",
                                    "password",
                                    "username",
                                    "phpinfo",
                                    "admin"
                                ]):
                                    finding = SecurityFinding(
                                        id=f"sensitive_endpoint_{endpoint}",
                                        title="Sensitive Information Disclosure",
                                        description=f"Accessible endpoint '{endpoint}' may disclose sensitive information",
                                        severity=VulnerabilitySeverity.MEDIUM,
                                        category=SecurityTestCategory.VULNERABILITY_SCANNING,
                                        affected_url=url,
                                        remediation="Restrict access to sensitive endpoints and remove information disclosure"
                                    )
                                    findings.append(finding)
                                    logger.warning(f"Sensitive endpoint accessible: {url}")
                                    
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout accessing {url}")
                    except Exception as e:
                        logger.debug(f"Error scanning {url}: {str(e)}")
                
                # Check for missing security headers
                common_headers = await self.check_security_headers(session)
                findings.extend(common_headers)
                
        except Exception as e:
            logger.error(f"Error in vulnerability scanning: {str(e)}")
        
        duration = time.time() - start_time
        
        result = SecurityTestResult(
            test_name="Common Vulnerabilities Scan",
            category=SecurityTestCategory.VULNERABILITY_SCANNING,
            passed=len(findings) == 0,
            findings=findings,
            duration=duration,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        self.findings.extend(findings)
        
        return result
    
    async def check_security_headers(self, session: aiohttp.ClientSession) -> List[SecurityFinding]:
        """Check for missing security headers."""
        findings = []
        
        try:
            async with session.get(self.base_url) as response:
                headers = response.headers
                
                # Security headers to check
                security_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY or SAMEORIGIN",
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=31536000",
                    "Content-Security-Policy": "Content Security Policy",
                    "Referrer-Policy": "strict-origin-when-cross-origin"
                }
                
                for header, description in security_headers.items():
                    if header not in headers:
                        severity = VulnerabilitySeverity.LOW
                        if header in ["X-Content-Type-Options", "X-Frame-Options"]:
                            severity = VulnerabilitySeverity.MEDIUM
                        
                        finding = SecurityFinding(
                            id=f"missing_header_{header}",
                            title=f"Missing Security Header: {header}",
                            description=f"Missing {header} header reduces security",
                            severity=severity,
                            category=SecurityTestCategory.VULNERABILITY_SCANNING,
                            remediation=f"Add {header} header with appropriate value"
                        )
                        findings.append(finding)
                        logger.warning(f"Missing security header: {header}")
        
        except Exception as e:
            logger.error(f"Error checking security headers: {str(e)}")
        
        return findings
    
    def _count_findings_by_severity(self) -> Dict[str, int]:
        """Count findings by severity."""
        counts = {severity.value: 0 for severity in VulnerabilitySeverity}
        
        for finding in self.findings:
            counts[finding.severity.value] += 1
        
        return counts
    
    def _count_findings_by_category(self) -> Dict[str, int]:
        """Count findings by category."""
        counts = {category.value: 0 for category in SecurityTestCategory}
        
        for finding in self.findings:
            counts[finding.category.value] += 1
        
        return counts
    
    def _calculate_security_score(self, results: List[SecurityTestResult]) -> float:
        """Calculate overall security score (0-100)."""
        if not results:
            return 0
        
        # Start with 100 points
        score = 100
        
        # Deduct points based on findings
        for finding in self.findings:
            if finding.severity == VulnerabilitySeverity.CRITICAL:
                score -= 20
            elif finding.severity == VulnerabilitySeverity.HIGH:
                score -= 15
            elif finding.severity == VulnerabilitySeverity.MEDIUM:
                score -= 10
            elif finding.severity == VulnerabilitySeverity.LOW:
                score -= 5
        
        return max(0, score)
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Analyze findings and generate recommendations
        critical_findings = [f for f in self.findings if f.severity == VulnerabilitySeverity.CRITICAL]
        high_findings = [f for f in self.findings if f.severity == VulnerabilitySeverity.HIGH]
        
        if critical_findings:
            recommendations.append("CRITICAL: Address critical security vulnerabilities immediately")
        
        if high_findings:
            recommendations.append("HIGH: Address high-severity security vulnerabilities promptly")
        
        # Specific recommendations based on findings
        auth_findings = [f for f in self.findings if f.category == SecurityTestCategory.AUTHENTICATION]
        if auth_findings:
            recommendations.append("Strengthen authentication mechanisms and implement multi-factor authentication")
        
        input_findings = [f for f in self.findings if f.category == SecurityTestCategory.INPUT_VALIDATION]
        if input_findings:
            recommendations.append("Implement proper input validation and sanitization across all user inputs")
        
        network_findings = [f for f in self.findings if f.category == SecurityTestCategory.NETWORK_SECURITY]
        if network_findings:
            recommendations.append("Review and strengthen network security configurations")
        
        # General recommendations
        recommendations.extend([
            "Implement regular security testing and code reviews",
            "Keep all software and dependencies up to date",
            "Implement monitoring and logging for security events",
            "Establish incident response procedures"
        ])
        
        return recommendations
    
    def _result_to_dict(self, result: SecurityTestResult) -> Dict[str, Any]:
        """Convert SecurityTestResult to dictionary."""
        return {
            "test_name": result.test_name,
            "category": result.category.value,
            "passed": result.passed,
            "findings_count": len(result.findings),
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "severity": f.severity.value,
                    "description": f.description,
                    "remediation": f.remediation
                }
                for f in result.findings
            ]
        }
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current security test status."""
        return {
            "total_tests_run": len(self.test_results),
            "total_findings": len(self.findings),
            "critical_findings": len([f for f in self.findings if f.severity == VulnerabilitySeverity.CRITICAL]),
            "high_findings": len([f for f in self.findings if f.severity == VulnerabilitySeverity.HIGH]),
            "test_categories": list(set(r.category.value for r in self.test_results))
        }


# Global security test suite instance
security_test_suite = SecurityTestSuite()