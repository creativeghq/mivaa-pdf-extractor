#!/bin/bash

# For each route file, extract @router decorators and following async def with their parameters
# to determine auth requirements

for file in app/api/*.py app/api/**/*.py; do
    [ ! -f "$file" ] && continue
    [ "$(basename $file)" = "__init__.py" ] && continue
    
    echo "=== $(basename $file) ==="
    
    # Use awk to capture @router decorator + next async def line
    awk '
    /@router\.(get|post|put|delete|patch)/ {
        route_line = $0
        method = ""
        path = ""
        
        # Extract method and path
        if ($0 ~ /@router\.get\(/) { method = "GET"; path = gensub(/.*@router\.get\("([^"]+)".*/,"\1","g") }
        else if ($0 ~ /@router\.post\(/) { method = "POST"; path = gensub(/.*@router\.post\("([^"]+)".*/,"\1","g") }
        else if ($0 ~ /@router\.put\(/) { method = "PUT"; path = gensub(/.*@router\.put\("([^"]+)".*/,"\1","g") }
        else if ($0 ~ /@router\.delete\(/) { method = "DELETE"; path = gensub(/.*@router\.delete\("([^"]+)".*/,"\1","g") }
        else if ($0 ~ /@router\.patch\(/) { method = "PATCH"; path = gensub(/.*@router\.patch\("([^"]+)".*/,"\1","g") }
        
        # Get next async def line
        getline
        while ($0 !~ /^async def/ && NF > 0) getline
        
        if ($0 ~ /^async def/) {
            handler = gensub(/async def ([a-zA-Z_0-9]+)\(.*/, "\1", "g")
            rest = $0
            
            # Print route info
            print method " " path
            
            # Check if there are Depends() calls
            if (rest ~ /Depends\(/) {
                print "  AUTH: " substr(rest, RSTART)
            } else {
                # Continue reading lines until we find params with Depends or closing paren
                while ($0 !~ /\):/) {
                    getline
                    if ($0 ~ /Depends\(/) {
                        print "  AUTH: " $0
                        break
                    }
                }
                if ($0 !~ /Depends\(/) print "  AUTH: NONE"
            }
            print ""
        }
    }
    ' "$file"
done
