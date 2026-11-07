#!/usr/bin/env bash
# Export slides/index.html to slides/prosumer-sentinel.pdf (16:9, one slide per page).
set -euo pipefail
cd "$(dirname "$0")"
google-chrome --headless=new --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$PWD/prosumer-sentinel.pdf" "file://$PWD/index.html"
