#!/usr/bin/env python3
"""
Check Hugging Face Endpoint Status Script

This script checks the status of all Hugging Face inference endpoints
and provides detailed information about their current state.

Usage:
    python3 scripts/check_endpoint_status.py
"""

import os
import sys
from datetime import datetime
from huggingface_hub import get_inference_endpoint

# HuggingFace token
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HUGGING_FACE_ACCESS_TOKEN')
if not HF_TOKEN:
    print("❌ ERROR: HUGGINGFACE_API_KEY not set in environment")
    sys.exit(1)

NAMESPACE = "basiliskan"

ENDPOINTS = [
    {
        "name": "mh-qwen332binstruct",
        "description": "Qwen Vision Model",
        "purpose": "Image analysis and classification",
        "url": "https://gbz6krk3i2is85b0.us-east-1.aws.endpoints.huggingface.cloud"
    },
    {
        "name": "mh-slig",
        "description": "SLIG/SigLIP2",
        "purpose": "Visual embeddings (768D)",
        "url": "https://f4kbl5do4tz6svct.us-east-1.aws.endpoints.huggingface.cloud"
    },
    {
        "name": "mh-yolo",
        "description": "YOLO DocParser",
        "purpose": "PDF layout detection",
        "url": "https://f763mkb5o68lmwtu.us-east-1.aws.endpoints.huggingface.cloud"
    }
]


def check_endpoint(endpoint_config):
    """Check status of a single endpoint."""
    name = endpoint_config["name"]
    description = endpoint_config["description"]
    purpose = endpoint_config["purpose"]
    url = endpoint_config["url"]
    
    try:
        endpoint = get_inference_endpoint(name, namespace=NAMESPACE, token=HF_TOKEN)
        endpoint.fetch()
        
        status = endpoint.status
        status_icon = "✅" if status == "running" else "⏸️" if status in ["paused", "scaledToZero"] else "⚠️"
        
        return {
            "name": name,
            "description": description,
            "purpose": purpose,
            "url": url,
            "status": status,
            "status_icon": status_icon,
            "error": None
        }
            
    except Exception as e:
        return {
            "name": name,
            "description": description,
            "purpose": purpose,
            "url": url,
            "status": "error",
            "status_icon": "❌",
            "error": str(e)
        }


def main():
    """Check all endpoints and display status."""
    print("=" * 80)
    print(f"🔍 Hugging Face Endpoint Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Namespace: {NAMESPACE}")
    print(f"Token: {HF_TOKEN[:10]}...")
    print()
    
    results = []
    for endpoint_config in ENDPOINTS:
        result = check_endpoint(endpoint_config)
        results.append(result)
    
    # Display results
    for result in results:
        print(f"{result['status_icon']} {result['name']}")
        print(f"   Description: {result['description']}")
        print(f"   Purpose: {result['purpose']}")
        print(f"   Status: {result['status']}")
        print(f"   URL: {result['url']}")
        if result['error']:
            print(f"   Error: {result['error']}")
        print()
    
    # Summary
    print("=" * 80)
    running = sum(1 for r in results if r['status'] == 'running')
    paused = sum(1 for r in results if r['status'] in ['paused', 'scaledToZero'])
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"📊 Summary: {running} running, {paused} paused, {errors} errors")
    
    if paused > 0:
        print("\n⚠️  Some endpoints are paused. Run 'python3 scripts/resume_all_endpoints.py' to resume them.")
    
    if errors > 0:
        print("\n❌ Some endpoints have errors. Check the error messages above.")
        return 1
    
    if running == len(results):
        print("\n✅ All endpoints are running!")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

