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

# Check if the file size is a multiple of 4 (32-bit values)
file_size=$(wc -c < "$1")
if [ $((file_size % 4)) -ne 0 ]; then
    echo "File size must be a multiple of 4 bytes"
    exit 1
fi

# Function to reverse the byte order of a 32-bit word
reverse_bytes() {
    local word=$1
    echo -n "${word:6:2}${word:4:2}${word:2:2}${word:0:2}"
}

# Convert endianness
for (( i=0; i<file_size; i+=4 )); do
    # Read 4 bytes
    bytes=$(dd bs=1 skip=$i count=4 if="$1" 2>/dev/null | xxd -p -c4)
    # Reverse bytes
    reversed=$(reverse_bytes "$bytes")
    # Output reversed bytes
    echo -n $reversed | xxd -r -p
done
