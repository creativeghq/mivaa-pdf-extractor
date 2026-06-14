#!/usr/bin/env python3
import re
from pathlib import Path

def extract_route_info(filepath):
    """Extract all routes from a FastAPI router file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    routes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for @router.METHOD decorators
        if '@router.' in line and '(' in line:
            match = re.search(r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', line)
            if match:
                method = match.group(1).upper()
                path = match.group(2)
                
                # Get the next async def
                j = i + 1
                while j < len(lines) and 'async def' not in lines[j]:
                    j += 1
                
                if j < len(lines):
                    # Read function signature (could span multiple lines)
                    sig_lines = []
                    k = j
                    while k < len(lines) and ')' not in lines[k]:
                        sig_lines.append(lines[k])
                        k += 1
                    if k < len(lines):
                        sig_lines.append(lines[k])  # Add the line with )
                    
                    sig = ' '.join(sig_lines)
                    
                    # Extract function name
                    func_match = re.search(r'async def (\w+)\(', sig)
                    func_name = func_match.group(1) if func_match else 'unknown'
                    
                    # Determine auth type
                    auth = "NONE"
                    if "get_workspace_context" in sig or "get_current_user" in sig or "require_" in sig:
                        auth = "JWT_REQUIRED"
                    elif "get_optional" in sig:
                        auth = "OPTIONAL_AUTH"
                    elif "authenticate_api_key" in sig:
                        auth = "API_KEY_BEARER"
                    
                    # Check for cron bypass (look ahead in function body)
                    body_lines = '\n'.join(lines[k:min(k+50, len(lines))])
                    if "x-cron-secret" in body_lines or "CRON_SECRET" in body_lines:
                        if auth == "JWT_REQUIRED":
                            auth = "JWT_WITH_CRON_BYPASS"
                        elif auth == "NONE":
                            auth = "CRON_SECRET_ONLY"
                    
                    routes.append({
                        'method': method,
                        'path': path,
                        'func': func_name,
                        'auth': auth,
                        'line': i + 1
                    })
        
        i += 1
    
    return routes

# Files to audit
key_files = [
    "app/api/rag_routes.py",
    "app/api/catalog_routes.py",
    "app/api/price_monitoring_routes.py",
    "app/api/mention_monitoring_routes.py",
    "app/api/job_research_routes.py",
    "app/api/project_tracking_routes.py",
    "app/api/interior_design_routes.py",
    "app/api/seo_agent_routes.py",
    "app/api/agent_routes.py",
]

all_routes = []
for fpath in key_files:
    if Path(fpath).exists():
        fname = Path(fpath).name
        routes = extract_route_info(fpath)
        for r in routes:
            r['file'] = fname
            all_routes.append(r)

# Group by auth type and output
by_auth = {}
for r in all_routes:
    auth = r['auth']
    if auth not in by_auth:
        by_auth[auth] = []
    by_auth[auth].append(r)

# Print report
print("\n" + "="*100)
print("FASTAPI ROUTE AUTHENTICATION AUDIT")
print("="*100)

for auth_type in ["JWT_REQUIRED", "JWT_WITH_CRON_BYPASS", "CRON_SECRET_ONLY", "API_KEY_BEARER", "OPTIONAL_AUTH", "NONE"]:
    if auth_type in by_auth:
        print(f"\n{auth_type} ({len(by_auth[auth_type])} routes)")
        print("-" * 100)
        for r in sorted(by_auth[auth_type], key=lambda x: (x['file'], x['line'])):
            print(f"{r['method']:6} {r['path']:50} | {r['file']:30} line {r['line']}")

# Find suspects
print("\n" + "="*100)
print("SUSPECTS: Routes that spawn background jobs but are JWT_REQUIRED without cron bypass")
print("="*100)

suspects = []
for r in all_routes:
    if r['auth'] == "JWT_REQUIRED":
        # Check if function name or path suggests background/cron/internal work
        keywords = ['cron', 'background', 'refresh', 'resume', 'restart', 'internal', 'job', 'monitor']
        if any(kw in r['func'].lower() or kw in r['path'].lower() for kw in keywords):
            suspects.append(r)

if suspects:
    for r in sorted(suspects, key=lambda x: (x['file'], x['line'])):
        print(f"{r['method']:6} {r['path']:50} | {r['file']:30} line {r['line']} | {r['func']}")
else:
    print("(No suspects found)")

