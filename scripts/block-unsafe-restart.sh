#!/bin/bash

# Block Unsafe Restart Script
# This script replaces direct systemctl restart commands
# Install as alias in .bashrc to prevent accidental restarts

echo "❌ Direct systemctl restart is blocked for mivaa-pdf-extractor"
echo ""
echo "Use the safe restart script instead:"
echo "  cd /var/www/mivaa-pdf-extractor"
echo "  bash scripts/safe-restart.sh"
echo ""
echo "Or for emergency force restart:"
echo "  bash scripts/safe-restart.sh --force --reason \"your reason\""
echo ""

exit 1

