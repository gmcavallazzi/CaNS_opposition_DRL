#!/bin/bash

# Script root directory (where the script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean files in data directory
echo "Cleaning files in data directory..."
cd "${SCRIPT_DIR}/data" && rm -f ve* pre* *out *.lock 2>/dev/null || true

# Back to main directory and clean remaining files
cd "${SCRIPT_DIR}" || exit 1
rm -f training*.csv 2>/dev/null || true
rm -f R-* 2>/dev/null || true
rm -f step_timing.txt 2>/dev/null || true

# Clean directories if they exist
for dir in tensorboard_logs checkpoints saved_models; do
    if [ -d "$dir" ]; then
        rm -rf "${dir:?}"/* 2>/dev/null || true
    fi
done

echo "Cleanup completed!"
