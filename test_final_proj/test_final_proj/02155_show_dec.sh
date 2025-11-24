#!/bin/bash

# Check if a file name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "File not found: $1"
    exit 1
fi

# Check if the file has exactly 128 bytes using wc -c
file_size=$(wc -c < "$1")
if [ "$file_size" -ne 128 ]; then
    echo "File must contain exactly 128 bytes"
    exit 1
fi

# Read the file byte by byte in little endian format
bytes=($(od -An -tx1 -v "$1"))

# Process and print the registers
for (( i = 0; i < 128; i += 4 )); do
    # Extract 4 bytes for the register (little endian)
    reg_bytes="${bytes[i+3]}${bytes[i+2]}${bytes[i+1]}${bytes[i]}"

    # Convert to signed 32-bit integer
    reg_value=$((16#${reg_bytes}))
    if (( reg_value >= 0x80000000 )); then
        ((reg_value -= 0x100000000))
    fi

    # Print the register value with minimized spacing
    printf "%-4s = %-11d" "x$((i / 4))" $reg_value

    # Print a newline every 4 registers, no extra line spacing
    if (( (i / 4 + 1) % 4 == 0 )); then
        echo
    else
        echo -n ', '
    fi
done
