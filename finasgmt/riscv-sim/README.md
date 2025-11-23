# RISC-V Simulator Overview

This crate implements a minimal RV32I simulator used for the final assignment. It models the architectural state (32 registers
and a program counter), 1 MB of byte-addressable memory sized to fit both programs and stack, and a loop that fetches,
decodes, and executes instructions until the program exits.

## Components
- **CPU state**: `Cpu` keeps `regs[32]` plus `pc`, enforcing that writes to `x0` are ignored.
- **Memory model**: `Memory` stores the input program and exposes byte/halfword/word load–store helpers with bounds checks.
- **Simulator**: `Simulator::run` executes in a loop, decoding opcodes and immediates, updating registers and memory,
and optionally dumping state after each instruction when `--trace` is passed.
- **I/O**: The CLI reads a binary program, runs it, then writes a 32 × 4-byte register dump (default `regdump.bin`) and prints
final register values and an exit reason.

## Supported instructions
The decoder/executor covers the RV32I base instructions used by the provided tests:
- Upper-immediates: `lui`, `auipc`
- Control flow: `jal`, `jalr`, and branches (`beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`)
- Loads/stores: `lb`, `lh`, `lw`, `lbu`, `lhu`, `sb`, `sh`, `sw`
- ALU ops: immediate and register forms for add/sub, shifts (logical/arithmetic), set-less-than (signed/unsigned), xor, or,
and
- System: `ecall` (exit when `a7 == 10`), with `ebreak` and other CSR ops reported as unsupported

## Usage
```
cargo run -- <program.bin> [output_dump] [--trace]
```
If `output_dump` is omitted, registers are written to `regdump.bin`. Use `--trace` to log each executed instruction and register
state to stdout.

## Testing
The simulator was validated against the provided task binaries and the unit tests shipped with the crate:
```
cargo test
cargo run -- ../tests/task1/addpos.bin /tmp/addpos.out && cmp /tmp/addpos.out ../tests/task1/addpos.res
cargo run -- ../tests/task2/branchmany.bin /tmp/branchmany.out && cmp /tmp/branchmany.out ../tests/task2/branchmany.res
cargo run -- ../tests/task3/width.bin /tmp/width.out && cmp /tmp/width.out ../tests/task3/width.res
```
These commands ensure arithmetic, control flow, and memory behaviors match the expected register dumps for tasks 1–3.
