#!/usr/bin/env python3
"""
Day Trading Orchestrator - Configuration Validator
Validates configuration files and detects common issues
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ValidationError:
    """Configuration validation error"""
    section: str
    field: str
    message: str
    severity: str  # "error", "warning", "info"

@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    infos: List[ValidationError]

class ConfigurationValidator:
    """Validates Day Trading Orchestrator configuration files"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            infos=[]
        )
    
    def validate(self) -> ValidationResult:
        """Run comprehensive configuration validation"""
        print(f"🔍 Validating configuration: {self.config_path}")
        
        # Load configuration file
        if not self.load_config():
            return self.result
        
        # Run validation checks
        self.check_file_structure()
        
        # Generate summary
        self.generate_summary()
        
        return self.result
    
    def load_config(self) -> bool:
        """Load configuration file"""
        try:
            if not self.config_path.exists():
                self.add_error("global", "config_file", "Configuration file not found", "error")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.add_info("global", "config_file", "Configuration file loaded successfully", "info")
            return True
            
        except json.JSONDecodeError as e:
            self.add_error("global", "json_syntax", f"Invalid JSON syntax: {e}", "error")
            return False
        except Exception as e:
            self.add_error("global", "file_load", f"Failed to load configuration: {e}", "error")
            return False
    
    def check_file_structure(self):
        """Check configuration file structure"""
        required_sections = ["database", "brokers", "risk"]
        
        for section in required_sections:
            if section not in self.config:
                self.add_error("global", f"missing_{section}", f"Required section '{section}' not found", "error")
            else:
                self.add_info("global", f"section_{section}", f"Section '{section}' found", "info")
    
    def add_error(self, section: str, field: str, message: str, severity: str = "error"):
        """Add validation error"""
        error = ValidationError(section, field, message, severity)
        
        if severity == "error":
            self.result.errors.append(error)
            self.result.is_valid = False
        elif severity == "warning":
            self.result.warnings.append(error)
        else:
            self.result.infos.append(error)
    
    def add_warning(self, section: str, field: str, message: str, severity: str = "warning"):
        """Add validation warning"""
        self.add_error(section, field, message, severity)
    
    def add_info(self, section: str, field: str, message: str, severity: str = "info"):
        """Add validation info"""
        self.add_error(section, field, message, severity)
    
    def generate_summary(self):
        """Generate validation summary"""
        print(f"\n{'='*60}")
        print("📋 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        if self.result.is_valid and not self.result.warnings:
            print("✅ Configuration is valid!")
        elif self.result.is_valid:
            print("⚠️  Configuration is valid with warnings")
        else:
            print("❌ Configuration has errors")
        
        print(f"\n📊 Results:")
        print(f"  Errors: {len(self.result.errors)}")
        print(f"  Warnings: {len(self.result.warnings)}")
        print(f"  Info: {len(self.result.infos)}")
        
        # Print errors
        if self.result.errors:
            print(f"\n❌ ERRORS ({len(self.result.errors)}):")
            for error in self.result.errors:
                print(f"  • {error.section}.{error.field}: {error.message}")
        
        # Print warnings
        if self.result.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.result.warnings)}):")
            for warning in self.result.warnings:
                print(f"  • {warning.section}.{warning.field}: {warning.message}")
    
    def save_report(self, filename: str = None):
        """Save validation report to file"""
        if not filename:
            timestamp = Path(self.config_path).stem
            filename = f"validation_report_{timestamp}.json"
        
        report = {
            "config_file": str(self.config_path),
            "is_valid": self.result.is_valid,
            "summary": {
                "errors": len(self.result.errors),
                "warnings": len(self.result.warnings),
                "infos": len(self.result.infos)
            },
            "errors": [self.error_to_dict(error) for error in self.result.errors],
            "warnings": [self.error_to_dict(error) for error in self.result.warnings],
            "infos": [self.error_to_dict(error) for error in self.result.infos]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Validation report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save validation report: {e}")
    
    def error_to_dict(self, error: ValidationError) -> Dict[str, str]:
        """Convert error to dictionary"""
        return {
            "section": error.section,
            "field": error.field,
            "message": error.message,
            "severity": error.severity
        }

def main():
    """Main validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Configuration Validator")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--errors-only", action="store_true", help="Show only errors")
    parser.add_argument("--warnings-only", action="store_true", help="show only warnings")
    
    args = parser.parse_args()
    
    # Validate configuration
    validator = ConfigurationValidator(args.config)
    result = validator.validate()
    
    # Save report if requested
    if args.output:
        validator.save_report(args.output)
    
    # Exit with appropriate code
    exit_code = 0 if result.is_valid else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()