#!/usr/bin/env python3
"""
MIVAA PDF Extractor Deployment Validation Script

This script validates that all required environment variables and secrets
are properly configured for deployment of the MIVAA PDF Extractor service.

Usage:
    python validate-deployment.py [--generate-template]

Options:
    --generate-template    Generate a .env template file with all variables
"""

import os
import sys
import argparse
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


class ValidationLevel(Enum):
    """Validation levels for environment variables"""
    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"


@dataclass
class EnvVar:
    """Environment variable definition"""
    name: str
    level: ValidationLevel
    description: str
    default: Optional[str] = None
    category: str = "General"


class DeploymentValidator:
    """Validates deployment configuration for MIVAA PDF Extractor"""
    
    def __init__(self):
        self.env_vars = self._define_environment_variables()
        self.validation_results = {}
        
    def _define_environment_variables(self) -> List[EnvVar]:
        """Define all environment variables used by MIVAA PDF Extractor"""
        return [
            # Core Application Configuration
            EnvVar("APP_NAME", ValidationLevel.OPTIONAL, "Application name", "PDF Processing Service", category="Core"),
            EnvVar("APP_VERSION", ValidationLevel.OPTIONAL, "Application version", "1.0.0", category="Core"),
            EnvVar("DEBUG", ValidationLevel.OPTIONAL, "Enable debug mode", "false", category="Core"),
            EnvVar("HOST", ValidationLevel.OPTIONAL, "Server host", "0.0.0.0", category="Core"),
            EnvVar("PORT", ValidationLevel.OPTIONAL, "Server port", "8000", category="Core"),
            EnvVar("API_PREFIX", ValidationLevel.OPTIONAL, "API route prefix", "/api/v1", category="Core"),
            EnvVar("CORS_ORIGINS", ValidationLevel.OPTIONAL, "Allowed CORS origins", "*", category="Core"),
            EnvVar("LOG_LEVEL", ValidationLevel.OPTIONAL, "Logging level", "INFO", category="Core"),
            EnvVar("MAX_FILE_SIZE", ValidationLevel.OPTIONAL, "Maximum upload file size", "104857600", category="Core"),
            EnvVar("MAX_WORKERS", ValidationLevel.OPTIONAL, "Maximum worker processes", "4", category="Core"),
            EnvVar("REQUEST_TIMEOUT", ValidationLevel.OPTIONAL, "Request timeout in seconds", "300", category="Core"),
            
            # Database & Storage Configuration
            EnvVar("SUPABASE_URL", ValidationLevel.REQUIRED, "Supabase project URL", category="Database"),
            EnvVar("SUPABASE_ANON_KEY", ValidationLevel.REQUIRED, "Supabase anonymous key", category="Database"),
            EnvVar("SUPABASE_SERVICE_ROLE_KEY", ValidationLevel.REQUIRED, "Supabase service role key", category="Database"),
            EnvVar("SUPABASE_STORAGE_BUCKET", ValidationLevel.OPTIONAL, "Supabase storage bucket name", "pdf-documents", category="Database"),
            EnvVar("DATABASE_POOL_SIZE", ValidationLevel.OPTIONAL, "Database connection pool size", "10", category="Database"),
            EnvVar("DATABASE_MAX_OVERFLOW", ValidationLevel.OPTIONAL, "Database max overflow connections", "20", category="Database"),
            EnvVar("DATABASE_TIMEOUT", ValidationLevel.OPTIONAL, "Database connection timeout", "30", category="Database"),
            
            # AI/ML Service Integration
            EnvVar("OPENAI_API_KEY", ValidationLevel.REQUIRED, "OpenAI API key for LLM and embeddings", category="AI/ML"),
            EnvVar("OPENAI_ORG_ID", ValidationLevel.OPTIONAL, "OpenAI organization ID", category="AI/ML"),
            EnvVar("OPENAI_PROJECT_ID", ValidationLevel.OPTIONAL, "OpenAI project ID", category="AI/ML"),
            EnvVar("ANTHROPIC_API_KEY", ValidationLevel.OPTIONAL, "Anthropic API key for Claude models", category="AI/ML"),
            EnvVar("HUGGINGFACE_API_TOKEN", ValidationLevel.REQUIRED, "Hugging Face API token for 3D generation models", category="AI/ML"),
            EnvVar("HUGGING_FACE_ACCESS_TOKEN", ValidationLevel.REQUIRED, "Alternative Hugging Face token (same as above)", category="AI/ML"),
            EnvVar("REPLICATE_API_KEY", ValidationLevel.REQUIRED, "Replicate API key for 3D generation models", category="AI/ML"),
            EnvVar("JINA_API_KEY", ValidationLevel.REQUIRED, "Jina AI API key for embeddings and material scraping", category="AI/ML"),
            EnvVar("FIRECRAWL_API_KEY", ValidationLevel.REQUIRED, "Firecrawl API key for web scraping", category="AI/ML"),
            EnvVar("LLAMAINDEX_EMBEDDING_MODEL", ValidationLevel.OPTIONAL, "LlamaIndex embedding model", "text-embedding-3-small", category="AI/ML"),
            EnvVar("LLAMAINDEX_LLM_MODEL", ValidationLevel.OPTIONAL, "LlamaIndex LLM model", "gpt-3.5-turbo", category="AI/ML"),
            EnvVar("LLAMAINDEX_CHUNK_SIZE", ValidationLevel.OPTIONAL, "Text chunk size for processing", "1024", category="AI/ML"),
            EnvVar("LLAMAINDEX_CHUNK_OVERLAP", ValidationLevel.OPTIONAL, "Text chunk overlap", "200", category="AI/ML"),
            EnvVar("LLAMAINDEX_SIMILARITY_TOP_K", ValidationLevel.OPTIONAL, "Top K similarity results", "5", category="AI/ML"),
            EnvVar("LLAMAINDEX_STORAGE_DIR", ValidationLevel.OPTIONAL, "LlamaIndex storage directory", "./data/llamaindex", category="AI/ML"),
            EnvVar("LLAMAINDEX_ENABLE_RAG", ValidationLevel.OPTIONAL, "Enable RAG functionality", "true", category="AI/ML"),
            
            # Multi-modal Processing Configuration
            EnvVar("ENABLE_MULTIMODAL", ValidationLevel.OPTIONAL, "Enable multi-modal processing", "true", category="Processing"),
            EnvVar("MULTIMODAL_LLM_MODEL", ValidationLevel.OPTIONAL, "Multi-modal LLM model", "gpt-4-vision-preview", category="Processing"),
            EnvVar("MULTIMODAL_MAX_TOKENS", ValidationLevel.OPTIONAL, "Maximum tokens for multi-modal", "4096", category="Processing"),
            EnvVar("MULTIMODAL_TEMPERATURE", ValidationLevel.OPTIONAL, "Temperature for multi-modal", "0.1", category="Processing"),
            EnvVar("MULTIMODAL_IMAGE_DETAIL", ValidationLevel.OPTIONAL, "Image detail level", "high", category="Processing"),
            EnvVar("MULTIMODAL_BATCH_SIZE", ValidationLevel.OPTIONAL, "Batch size for processing", "5", category="Processing"),
            EnvVar("MULTIMODAL_TIMEOUT", ValidationLevel.OPTIONAL, "Timeout for multi-modal requests", "60", category="Processing"),
            
            # OCR Processing Configuration
            EnvVar("OCR_ENABLED", ValidationLevel.OPTIONAL, "Enable OCR processing", "true", category="OCR"),
            EnvVar("OCR_LANGUAGE", ValidationLevel.OPTIONAL, "OCR language code", "en", category="OCR"),
            EnvVar("OCR_CONFIDENCE_THRESHOLD", ValidationLevel.OPTIONAL, "OCR confidence threshold", "0.6", category="OCR"),
            EnvVar("OCR_ENGINE", ValidationLevel.OPTIONAL, "OCR engine to use", "easyocr", category="OCR"),
            EnvVar("OCR_GPU_ENABLED", ValidationLevel.OPTIONAL, "Enable GPU for OCR", "false", category="OCR"),
            EnvVar("OCR_PREPROCESSING_ENABLED", ValidationLevel.OPTIONAL, "Enable OCR preprocessing", "true", category="OCR"),
            EnvVar("OCR_DESKEW_ENABLED", ValidationLevel.OPTIONAL, "Enable image deskewing", "true", category="OCR"),
            EnvVar("OCR_NOISE_REMOVAL_ENABLED", ValidationLevel.OPTIONAL, "Enable noise removal", "true", category="OCR"),
            
            # Image Processing Configuration
            EnvVar("IMAGE_PROCESSING_ENABLED", ValidationLevel.OPTIONAL, "Enable image processing", "true", category="Image"),
            EnvVar("IMAGE_ANALYSIS_MODEL", ValidationLevel.OPTIONAL, "Model for image analysis", "gpt-4-vision-preview", category="Image"),
            EnvVar("IMAGE_RESIZE_MAX_WIDTH", ValidationLevel.OPTIONAL, "Maximum image width", "2048", category="Image"),
            EnvVar("IMAGE_RESIZE_MAX_HEIGHT", ValidationLevel.OPTIONAL, "Maximum image height", "2048", category="Image"),
            EnvVar("IMAGE_COMPRESSION_QUALITY", ValidationLevel.OPTIONAL, "Image compression quality", "85", category="Image"),
            EnvVar("IMAGE_FORMAT_CONVERSION", ValidationLevel.OPTIONAL, "Target image format", "JPEG", category="Image"),
            EnvVar("DEFAULT_IMAGE_FORMAT", ValidationLevel.OPTIONAL, "Default image format", "png", category="Image"),
            EnvVar("DEFAULT_IMAGE_QUALITY", ValidationLevel.OPTIONAL, "Default image quality", "95", category="Image"),
            EnvVar("WRITE_IMAGES", ValidationLevel.OPTIONAL, "Write extracted images", "true", category="Image"),
            EnvVar("EXTRACT_TABLES", ValidationLevel.OPTIONAL, "Extract tables from PDFs", "true", category="Image"),
            EnvVar("EXTRACT_IMAGES", ValidationLevel.OPTIONAL, "Extract images from PDFs", "true", category="Image"),
            
            # Authentication & Security Configuration
            EnvVar("JWT_SECRET_KEY", ValidationLevel.REQUIRED, "JWT signing secret key", category="Security"),
            EnvVar("JWT_ALGORITHM", ValidationLevel.OPTIONAL, "JWT algorithm", "HS256", category="Security"),
            EnvVar("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", ValidationLevel.OPTIONAL, "Access token expiry", "30", category="Security"),
            EnvVar("JWT_REFRESH_TOKEN_EXPIRE_DAYS", ValidationLevel.OPTIONAL, "Refresh token expiry", "7", category="Security"),
            EnvVar("JWT_ISSUER", ValidationLevel.OPTIONAL, "JWT issuer", "material-kai-platform", category="Security"),
            EnvVar("JWT_AUDIENCE", ValidationLevel.OPTIONAL, "JWT audience", "mivaa-pdf-extractor", category="Security"),
            EnvVar("MAX_REQUESTS_PER_MINUTE", ValidationLevel.OPTIONAL, "Rate limiting", "60", category="Security"),
            
            # Material Kai Platform Integration
            EnvVar("MATERIAL_KAI_PLATFORM_URL", ValidationLevel.OPTIONAL, "Platform API URL", "https://api.materialkai.vision", category="Platform"),
            EnvVar("MATERIAL_KAI_API_KEY", ValidationLevel.REQUIRED, "Platform API key", category="Platform"),
            EnvVar("MATERIAL_KAI_WORKSPACE_ID", ValidationLevel.REQUIRED, "Platform workspace ID", category="Platform"),
            EnvVar("MATERIAL_KAI_SERVICE_NAME", ValidationLevel.OPTIONAL, "Service identifier", "mivaa-pdf-extractor", category="Platform"),
            EnvVar("MATERIAL_KAI_SYNC_ENABLED", ValidationLevel.OPTIONAL, "Enable platform sync", "true", category="Platform"),
            EnvVar("MATERIAL_KAI_REAL_TIME_ENABLED", ValidationLevel.OPTIONAL, "Enable real-time updates", "true", category="Platform"),
            EnvVar("MATERIAL_KAI_BATCH_SIZE", ValidationLevel.OPTIONAL, "Batch processing size", "10", category="Platform"),
            EnvVar("MATERIAL_KAI_RETRY_ATTEMPTS", ValidationLevel.OPTIONAL, "Retry attempts", "3", category="Platform"),
            EnvVar("MATERIAL_KAI_TIMEOUT", ValidationLevel.OPTIONAL, "Request timeout", "30", category="Platform"),
            
            # Monitoring & Error Tracking
            EnvVar("SENTRY_DSN", ValidationLevel.OPTIONAL, "Sentry DSN for error tracking", category="Monitoring"),
            EnvVar("SENTRY_ENVIRONMENT", ValidationLevel.OPTIONAL, "Sentry environment", "development", category="Monitoring"),
            EnvVar("SENTRY_TRACES_SAMPLE_RATE", ValidationLevel.OPTIONAL, "Sentry traces sample rate", "0.1", category="Monitoring"),
            EnvVar("SENTRY_PROFILES_SAMPLE_RATE", ValidationLevel.OPTIONAL, "Sentry profiles sample rate", "0.1", category="Monitoring"),
            EnvVar("SENTRY_ENABLED", ValidationLevel.OPTIONAL, "Enable Sentry monitoring", "false", category="Monitoring"),
            EnvVar("SENTRY_RELEASE", ValidationLevel.OPTIONAL, "Sentry release version", category="Monitoring"),
            EnvVar("SENTRY_SERVER_NAME", ValidationLevel.OPTIONAL, "Sentry server name", category="Monitoring"),
            
            # Digital Ocean Deployment Configuration
            EnvVar("DO_API_TOKEN", ValidationLevel.REQUIRED, "Digital Ocean API token", category="Deployment"),
            EnvVar("DO_DROPLET_NAME", ValidationLevel.REQUIRED, "Droplet name for deployment", category="Deployment"),
            EnvVar("DO_REGION", ValidationLevel.OPTIONAL, "Digital Ocean region", "nyc3", category="Deployment"),
            EnvVar("DO_SIZE", ValidationLevel.OPTIONAL, "Droplet size", "s-2vcpu-4gb", category="Deployment"),
            EnvVar("DO_SSH_KEY_NAME", ValidationLevel.REQUIRED, "SSH key name for droplet access", category="Deployment"),
            EnvVar("DO_DOMAIN", ValidationLevel.OPTIONAL, "Domain for the application", category="Deployment"),
            
            # CI/CD Configuration
            EnvVar("GITHUB_TOKEN", ValidationLevel.REQUIRED, "GitHub token for CI/CD", category="CI/CD"),
            EnvVar("DOCKER_REGISTRY", ValidationLevel.OPTIONAL, "Docker registry URL", "ghcr.io", category="CI/CD"),
            EnvVar("DOCKER_IMAGE_NAME", ValidationLevel.OPTIONAL, "Docker image name", "mivaa-pdf-extractor", category="CI/CD"),
        ]
    
    def validate_environment(self) -> Dict[str, Dict]:
        """Validate all environment variables"""
        results = {}
        
        for category in self._get_categories():
            results[category] = {
                'variables': [],
                'required_missing': 0,
                'recommended_missing': 0,
                'total_missing': 0
            }
        
        for env_var in self.env_vars:
            value = os.getenv(env_var.name)
            status = self._get_status(env_var, value)
            
            var_result = {
                'name': env_var.name,
                'level': env_var.level.value,
                'description': env_var.description,
                'default': env_var.default,
                'value': value,
                'status': status,
                'configured': value is not None
            }
            
            results[env_var.category]['variables'].append(var_result)
            
            if not var_result['configured']:
                results[env_var.category]['total_missing'] += 1
                if env_var.level == ValidationLevel.REQUIRED:
                    results[env_var.category]['required_missing'] += 1
                elif env_var.level == ValidationLevel.RECOMMENDED:
                    results[env_var.category]['recommended_missing'] += 1
        
        self.validation_results = results
        return results
    
    def _get_categories(self) -> List[str]:
        """Get all unique categories"""
        return list(set(env_var.category for env_var in self.env_vars))
    
    def _get_status(self, env_var: EnvVar, value: Optional[str]) -> str:
        """Get validation status for an environment variable"""
        if value is None:
            if env_var.level == ValidationLevel.REQUIRED:
                return "❌ MISSING (REQUIRED)"
            elif env_var.level == ValidationLevel.RECOMMENDED:
                return "⚠️ MISSING (RECOMMENDED)"
            else:
                return "ℹ️ MISSING (OPTIONAL)"
        else:
            return "✅ CONFIGURED"
    
    def print_validation_report(self):
        """Print a detailed validation report"""
        if not self.validation_results:
            self.validate_environment()
        
        print("=" * 80)
        print("MIVAA PDF EXTRACTOR - DEPLOYMENT VALIDATION REPORT")
        print("=" * 80)
        print()
        
        total_required_missing = 0
        total_recommended_missing = 0
        total_configured = 0
        total_variables = len(self.env_vars)
        
        for category, data in self.validation_results.items():
            print(f"📂 {category.upper()} CONFIGURATION")
            print("-" * 50)
            
            total_required_missing += data['required_missing']
            total_recommended_missing += data['recommended_missing']
            
            for var in data['variables']:
                status_icon = var['status'].split()[0]
                print(f"  {status_icon} {var['name']:<35} | {var['level']:<12} | {var['description']}")
                if var['configured']:
                    total_configured += 1
                elif var['default']:
                    print(f"    └─ Default: {var['default']}")
            
            print()
            
            # Category summary
            missing_count = data['total_missing']
            configured_count = len(data['variables']) - missing_count
            print(f"  📊 Category Summary: {configured_count}/{len(data['variables'])} configured")
            if data['required_missing'] > 0:
                print(f"  🚨 Required missing: {data['required_missing']}")
            if data['recommended_missing'] > 0:
                print(f"  ⚠️ Recommended missing: {data['recommended_missing']}")
            print()
        
        # Overall summary
        print("=" * 80)
        print("📊 OVERALL VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Variables: {total_variables}")
        print(f"Configured: {total_configured}")
        print(f"Missing: {total_variables - total_configured}")
        print()
        
        if total_required_missing > 0:
            print(f"🚨 CRITICAL: {total_required_missing} required variables are missing!")
            print("   Deployment will fail without these variables.")
        else:
            print("✅ All required variables are configured!")
        
        if total_recommended_missing > 0:
            print(f"⚠️ WARNING: {total_recommended_missing} recommended variables are missing.")
            print("   Some features may not work as expected.")
        
        print()
        
        # Deployment readiness
        if total_required_missing == 0:
            print("🚀 DEPLOYMENT STATUS: READY")
            print("   All required environment variables are configured.")
        else:
            print("🛑 DEPLOYMENT STATUS: NOT READY")
            print("   Please configure all required environment variables before deploying.")
        
        print("=" * 80)
    
    def generate_env_template(self, filename: str = ".env.template"):
        """Generate a .env template file"""
        with open(filename, 'w') as f:
            f.write("# MIVAA PDF Extractor Environment Variables Template\n")
            f.write("# Generated by validate-deployment.py\n")
            f.write("#\n")
            f.write("# Copy this file to .env and fill in the required values\n")
            f.write("# Required variables are marked with (REQUIRED)\n")
            f.write("# Recommended variables are marked with (RECOMMENDED)\n")
            f.write("# Optional variables are marked with (OPTIONAL)\n")
            f.write("\n")
            
            current_category = None
            for env_var in sorted(self.env_vars, key=lambda x: (x.category, x.level.value, x.name)):
                if env_var.category != current_category:
                    f.write(f"\n# {env_var.category.upper()} CONFIGURATION\n")
                    f.write("# " + "=" * 50 + "\n")
                    current_category = env_var.category
                
                f.write(f"\n# {env_var.description} ({env_var.level.value})\n")
                if env_var.default:
                    f.write(f"# Default: {env_var.default}\n")
                
                if env_var.level == ValidationLevel.REQUIRED:
                    f.write(f"{env_var.name}=\n")
                else:
                    f.write(f"# {env_var.name}={env_var.default or ''}\n")
        
        print(f"✅ Environment template generated: {filename}")
        print(f"   Copy this file to .env and configure the required variables.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate MIVAA PDF Extractor deployment configuration"
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate a .env template file"
    )
    
    args = parser.parse_args()
    
    validator = DeploymentValidator()
    
    if args.generate_template:
        validator.generate_env_template()
        return
    
    # Run validation
    validator.validate_environment()
    validator.print_validation_report()
    
    # Exit with error code if required variables are missing
    total_required_missing = sum(
        data['required_missing'] 
        for data in validator.validation_results.values()
    )
    
    if total_required_missing > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()