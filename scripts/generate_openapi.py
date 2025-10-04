#!/usr/bin/env python3
"""
Generate OpenAPI Schema Export Script

This script exports the current FastAPI OpenAPI schema to static files
for documentation and integration purposes.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def generate_openapi_schema():
    """Generate and export the OpenAPI schema to static files."""
    
    try:
        # Add the app directory to the Python path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Import and create the FastAPI app
        from app.main import create_app
        app = create_app()
        
        # Get the OpenAPI schema
        openapi_schema = app.openapi()
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "docs" / "openapi"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        json_file = output_dir / "openapi.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
        
        # Export as JavaScript module
        js_file = output_dir / "openapi.js"
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(f"// MIVAA PDF Extractor - OpenAPI Schema\n")
            f.write(f"// Auto-generated from FastAPI application\n")
            f.write(f"// Generated on: {datetime.now().isoformat()}\n\n")
            f.write("export const openApiSchema = ")
            f.write(json.dumps(openapi_schema, indent=2, ensure_ascii=False))
            f.write(";\n\nexport default openApiSchema;\n")
        
        # Export as TypeScript module
        ts_file = output_dir / "openapi.ts"
        with open(ts_file, 'w', encoding='utf-8') as f:
            f.write(f"// MIVAA PDF Extractor - OpenAPI Schema\n")
            f.write(f"// Auto-generated from FastAPI application\n")
            f.write(f"// Generated on: {datetime.now().isoformat()}\n\n")
            f.write("export interface OpenAPISchema {\n")
            f.write("  openapi: string;\n")
            f.write("  info: {\n")
            f.write("    title: string;\n")
            f.write("    version: string;\n")
            f.write("    description?: string;\n")
            f.write("    [key: string]: any;\n")
            f.write("  };\n")
            f.write("  servers?: Array<{\n")
            f.write("    url: string;\n")
            f.write("    description?: string;\n")
            f.write("  }>;\n")
            f.write("  paths: { [key: string]: any };\n")
            f.write("  components?: { [key: string]: any };\n")
            f.write("  tags?: Array<{\n")
            f.write("    name: string;\n")
            f.write("    description?: string;\n")
            f.write("  }>;\n")
            f.write("  [key: string]: any;\n")
            f.write("}\n\n")
            f.write("export const openApiSchema: OpenAPISchema = ")
            f.write(json.dumps(openapi_schema, indent=2, ensure_ascii=False))
            f.write(";\n\nexport default openApiSchema;\n")
        
        # Generate summary
        paths_count = len(openapi_schema.get('paths', {}))
        tags_count = len(openapi_schema.get('tags', []))
        version = openapi_schema.get('info', {}).get('version', 'unknown')
        title = openapi_schema.get('info', {}).get('title', 'unknown')
        
        print(f"✅ OpenAPI schema exported successfully!")
        print(f"📄 JSON: {json_file}")
        print(f"📄 JavaScript: {js_file}")
        print(f"📄 TypeScript: {ts_file}")
        print(f"📊 Statistics:")
        print(f"   - API Title: {title}")
        print(f"   - API Version: {version}")
        print(f"   - API Endpoints: {paths_count}")
        print(f"   - API Categories: {tags_count}")
        
        return openapi_schema
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the MIVAA project root and dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating OpenAPI schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_openapi_schema()
