#!/bin/bash

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 filename line_number"
    exit 1
fi

filename=$1
line_number=$2

# Check if file exists
if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found"
    exit 1
fi

# Extract the specified line and count numbers, including scientific notation and decimals
# This regex matches:
# - Standard integers: \b\d+\b
# - Decimal numbers: \b\d*\.\d+\b
# - Scientific notation: \b\d*\.?\d+[Ee][+-]?\d+\b
count=$(sed -n "${line_number}p" "$filename" | grep -o -E '\b[+-]?([0-9]*[.])?[0-9]+([Ee][+-]?[0-9]+)?\b' | wc -l)

echo "Number of numbers found in line $line_number: $count"

# Also show the actual numbers found (for verification)
echo "Numbers found:"
sed -n "${line_number}p" "$filename" | grep -o -E '\b[+-]?([0-9]*[.])?[0-9]+([Ee][+-]?[0-9]+)?\b'
