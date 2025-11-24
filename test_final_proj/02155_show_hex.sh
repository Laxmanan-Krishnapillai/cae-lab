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

# Read the file byte by byte
bytes=($(od -An -tx1 -v "$1"))

# Process and print the registers
for (( i = 0; i < 128; i += 4 )); do
    # Extract 4 bytes for the register (little endian)
    reg_bytes=( "${bytes[@]:i:4}" )
    reg_bytes_reversed=( ${reg_bytes[3]} ${reg_bytes[2]} ${reg_bytes[1]} ${reg_bytes[0]} )

    # Convert to hexadecimal and print the register value
    printf "x%02d = 0x%02X%02X%02X%02X" $((i / 4)) 0x${reg_bytes_reversed[0]} 0x${reg_bytes_reversed[1]} 0x${reg_bytes_reversed[2]} 0x${reg_bytes_reversed[3]}

    # Print a newline every 4 registers, no extra line spacing
    if (( (i / 4 + 1) % 4 == 0 )); then
        echo
    else
        echo -n ', '
    fi
done
