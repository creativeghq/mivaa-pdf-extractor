#!/bin/bash
export SUPABASE_SERVICE_ROLE_KEY=$(cat /proc/$(pgrep -f "uvicorn app.main:app" | head -1)/environ | tr '\0' '\n' | grep SUPABASE_SERVICE_ROLE_KEY | cut -d'=' -f2)
cd /var/www/mivaa-pdf-extractor/test-scripts
node reset-database-complete.js
