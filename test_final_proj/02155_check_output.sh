#!/bin/bash

# Function to read and convert registers from a file
read_registers() {
    local file=$1
    local bytes=($(od -An -tx1 -v "$file"))
    local registers=()

    for (( i = 0; i < 128; i += 4 )); do
        local reg_bytes="${bytes[i+3]}${bytes[i+2]}${bytes[i+1]}${bytes[i]}"
        local reg_value=$((16#${reg_bytes}))
        if (( reg_value >= 0x80000000 )); then
            ((reg_value -= 0x100000000))
        fi
        registers+=($reg_value)
    done

    echo "${registers[@]}"
}

# Check if two file names are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <correct_file> <file_to_check>"
    exit 1
fi

# Check if the files exist
if [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "One or both files not found"
    exit 1
fi

# Read registers from both files
correct_values_str=$(read_registers "$1")
check_values_str=$(read_registers "$2")

# Convert string to array
correct_values=($correct_values_str)
check_values=($check_values_str)

# Compare the registers
all_correct=true
for (( i = 0; i < 32; i++ )); do
    if [ "${correct_values[i]}" -ne "${check_values[i]}" ]; then
        all_correct=false
        # Convert to hexadecimal for display
        correct_hex=$(printf '%08X' $((${correct_values[i]} & 0xFFFFFFFF)))
        check_hex=$(printf '%08X' $((${check_values[i]} & 0xFFFFFFFF)))
        printf "Register x%02d: Incorrect value. Expected 0x%s (%d), got 0x%s (%d)\n" \
               $i $correct_hex ${correct_values[i]} $check_hex ${check_values[i]}
    fi
done

# Output result
if $all_correct; then
    echo "All registers have the correct values."
else
    echo "Some registers have incorrect values."
fi
