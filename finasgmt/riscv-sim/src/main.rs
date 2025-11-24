use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;

const MEMORY_SIZE: usize = 1 << 20; // 1 MB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitReason {
    ProgramExhausted,
    Ecall,
}

struct Cpu {
    regs: [u32; 32],
    pc: u32,
}

impl Cpu {
    fn new() -> Self {
        Self {
            regs: [0; 32],
            pc: 0,
        }
    }

    fn read_reg(&self, idx: usize) -> u32 {
        self.regs[idx]
    }

    fn write_reg(&mut self, idx: usize, value: u32) {
        if idx != 0 {
            self.regs[idx] = value;
        }
    }
}

struct Memory {
    data: Vec<u8>,
}

impl Memory {
    fn new(program: &[u8]) -> Self {
        let mut size = MEMORY_SIZE;
        if program.len() > size {
            size = program.len();
        }
        let mut data = vec![0u8; size];
        data[..program.len()].copy_from_slice(program);
        Self { data }
    }

    fn load_word(&self, addr: u32) -> Result<u32, String> {
        let addr = addr as usize;
        if addr + 4 > self.data.len() {
            return Err(format!("Load out of bounds at address 0x{addr:x}"));
        }
        let bytes = &self.data[addr..addr + 4];
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn load_byte(&self, addr: u32) -> Result<u8, String> {
        let addr = addr as usize;
        self.data
            .get(addr)
            .copied()
            .ok_or_else(|| format!("Load out of bounds at address 0x{addr:x}"))
    }

    fn load_half(&self, addr: u32) -> Result<u16, String> {
        let addr = addr as usize;
        if addr + 2 > self.data.len() {
            return Err(format!("Load out of bounds at address 0x{addr:x}"));
        }
        let bytes = &self.data[addr..addr + 2];
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn store_byte(&mut self, addr: u32, value: u8) -> Result<(), String> {
        let addr = addr as usize;
        if addr >= self.data.len() {
            return Err(format!("Store out of bounds at address 0x{addr:x}"));
        }
        self.data[addr] = value;
        Ok(())
    }

    fn store_half(&mut self, addr: u32, value: u16) -> Result<(), String> {
        let addr = addr as usize;
        if addr + 2 > self.data.len() {
            return Err(format!("Store out of bounds at address 0x{addr:x}"));
        }
        self.data[addr..addr + 2].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn store_word(&mut self, addr: u32, value: u32) -> Result<(), String> {
        let addr = addr as usize;
        if addr + 4 > self.data.len() {
            return Err(format!("Store out of bounds at address 0x{addr:x}"));
        }
        self.data[addr..addr + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}

struct Simulator {
    cpu: Cpu,
    memory: Memory,
    program_len: u32,
    trace: bool,
}

impl Simulator {
    fn new(program: Vec<u8>, trace: bool) -> Self {
        let program_len = program.len() as u32;
        Self {
            cpu: Cpu::new(),
            memory: Memory::new(&program),
            program_len,
            trace,
        }
    }

    fn run(&mut self) -> Result<ExitReason, String> {
        loop {
            if self.cpu.pc >= self.program_len {
                return Ok(ExitReason::ProgramExhausted);
            }

            let instr = self.memory.load_word(self.cpu.pc)?;
            let old_pc = self.cpu.pc;
            let mut next_pc = self.cpu.pc.wrapping_add(4);

            let opcode = instr & 0x7f;
            let rd = ((instr >> 7) & 0x1f) as usize;
            let funct3 = (instr >> 12) & 0x7;
            let rs1 = ((instr >> 15) & 0x1f) as usize;
            let rs2 = ((instr >> 20) & 0x1f) as usize;
            let funct7 = (instr >> 25) & 0x7f;

            match opcode {
                0x37 => {
                    // LUI
                    let imm = instr & 0xfffff000;
                    self.cpu.write_reg(rd, imm);
                }
                0x17 => {
                    // AUIPC
                    let imm = instr & 0xfffff000;
                    self.cpu.write_reg(rd, self.cpu.pc.wrapping_add(imm));
                }
                0x6f => {
                    // JAL
                    let imm = decode_j_imm(instr);
                    self.cpu.write_reg(rd, self.cpu.pc.wrapping_add(4));
                    next_pc = ((self.cpu.pc as i64) + imm as i64) as u32;
                }
                0x67 => {
                    // JALR
                    let imm = decode_i_imm(instr);
                    let target = (self.cpu.read_reg(rs1) as i64 + imm as i64) as u32 & !1;
                    self.cpu.write_reg(rd, self.cpu.pc.wrapping_add(4));
                    next_pc = target;
                }
                0x63 => {
                    // Branches
                    let imm = decode_b_imm(instr);
                    let take_branch = match funct3 {
                        0x0 => self.cpu.read_reg(rs1) == self.cpu.read_reg(rs2),           // BEQ
                        0x1 => self.cpu.read_reg(rs1) != self.cpu.read_reg(rs2),           // BNE
                        0x4 => (self.cpu.read_reg(rs1) as i32) < (self.cpu.read_reg(rs2) as i32), // BLT
                        0x5 => (self.cpu.read_reg(rs1) as i32) >= (self.cpu.read_reg(rs2) as i32), // BGE
                        0x6 => self.cpu.read_reg(rs1) < self.cpu.read_reg(rs2),            // BLTU
                        0x7 => self.cpu.read_reg(rs1) >= self.cpu.read_reg(rs2),           // BGEU
                        _ => return Err(format!("Unknown branch funct3: 0x{funct3:x}")),
                    };
                    if take_branch {
                        next_pc = ((self.cpu.pc as i64) + imm as i64) as u32;
                    }
                }
                0x03 => {
                    // Loads
                    let imm = decode_i_imm(instr);
                    let addr = (self.cpu.read_reg(rs1) as i64 + imm as i64) as u32;
                    match funct3 {
                        0x0 => {
                            // LB
                            let byte = self.memory.load_byte(addr)? as i8 as i32 as u32;
                            self.cpu.write_reg(rd, byte);
                        }
                        0x1 => {
                            // LH
                            let half = self.memory.load_half(addr)? as i16 as i32 as u32;
                            self.cpu.write_reg(rd, half);
                        }
                        0x2 => {
                            // LW
                            let word = self.memory.load_word(addr)?;
                            self.cpu.write_reg(rd, word);
                        }
                        0x4 => {
                            // LBU
                            let byte = self.memory.load_byte(addr)? as u32;
                            self.cpu.write_reg(rd, byte);
                        }
                        0x5 => {
                            // LHU
                            let half = self.memory.load_half(addr)? as u32;
                            self.cpu.write_reg(rd, half);
                        }
                        _ => return Err(format!("Unknown load funct3: 0x{funct3:x}")),
                    }
                }
                0x23 => {
                    // Stores
                    let imm = decode_s_imm(instr);
                    let addr = (self.cpu.read_reg(rs1) as i64 + imm as i64) as u32;
                    match funct3 {
                        0x0 => self.memory.store_byte(addr, (self.cpu.read_reg(rs2) & 0xff) as u8)?, // SB
                        0x1 => self
                            .memory
                            .store_half(addr, (self.cpu.read_reg(rs2) & 0xffff) as u16)?, // SH
                        0x2 => self.memory.store_word(addr, self.cpu.read_reg(rs2))?, // SW
                        _ => return Err(format!("Unknown store funct3: 0x{funct3:x}")),
                    }
                }
                0x13 => {
                    // Immediate arithmetic
                    let imm = decode_i_imm(instr);
                    match funct3 {
                        0x0 => self.cpu.write_reg(rd, self.cpu.read_reg(rs1).wrapping_add(imm as u32)), // ADDI
                        0x2 => self.cpu.write_reg(
                            rd,
                            if (self.cpu.read_reg(rs1) as i32) < imm { 1 } else { 0 },
                        ), // SLTI
                        0x3 => self.cpu.write_reg(
                            rd,
                            if self.cpu.read_reg(rs1) < imm as u32 { 1 } else { 0 },
                        ), // SLTIU
                        0x4 => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) ^ imm as u32), // XORI
                        0x6 => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) | imm as u32), // ORI
                        0x7 => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) & imm as u32), // ANDI
                        0x1 => {
                            // SLLI
                            let shamt = (imm as u32) & 0x1f;
                            self.cpu.write_reg(rd, self.cpu.read_reg(rs1) << shamt);
                        }
                        0x5 => {
                            let shamt = (imm as u32) & 0x1f;
                            if funct7 == 0x00 {
                                // SRLI
                                self.cpu.write_reg(rd, self.cpu.read_reg(rs1) >> shamt);
                            } else if funct7 == 0x20 {
                                // SRAI
                                self.cpu.write_reg(rd, ((self.cpu.read_reg(rs1) as i32) >> shamt) as u32);
                            } else {
                                return Err(format!("Unknown shift immediate funct7: 0x{funct7:x}"));
                            }
                        }
                        _ => return Err(format!("Unknown immediate funct3: 0x{funct3:x}")),
                    }
                }
                0x33 => {
                    // Register arithmetic
                    match (funct3, funct7) {
                        (0x0, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1).wrapping_add(self.cpu.read_reg(rs2))), // ADD
                        (0x0, 0x20) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1).wrapping_sub(self.cpu.read_reg(rs2))), // SUB
                        (0x1, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) << (self.cpu.read_reg(rs2) & 0x1f)), // SLL
                        (0x2, 0x00) => self.cpu.write_reg(
                            rd,
                            if (self.cpu.read_reg(rs1) as i32) < (self.cpu.read_reg(rs2) as i32) {
                                1
                            } else {
                                0
                            },
                        ), // SLT
                        (0x3, 0x00) => self.cpu.write_reg(rd, if self.cpu.read_reg(rs1) < self.cpu.read_reg(rs2) { 1 } else { 0 }), // SLTU
                        (0x4, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) ^ self.cpu.read_reg(rs2)), // XOR
                        (0x5, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) >> (self.cpu.read_reg(rs2) & 0x1f)), // SRL
                        (0x5, 0x20) => self.cpu.write_reg(
                            rd,
                            ((self.cpu.read_reg(rs1) as i32) >> (self.cpu.read_reg(rs2) & 0x1f)) as u32,
                        ), // SRA
                        (0x6, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) | self.cpu.read_reg(rs2)), // OR
                        (0x7, 0x00) => self.cpu.write_reg(rd, self.cpu.read_reg(rs1) & self.cpu.read_reg(rs2)), // AND
                        _ => return Err(format!("Unknown register operation funct3 0x{funct3:x} funct7 0x{funct7:x}")),
                    }
                }
                0x73 => {
                    // SYSTEM
                    if instr == 0x00000073 {
                        // ECALL
                        if self.cpu.read_reg(17) == 10 {
                            self.cpu.pc = next_pc;
                            if self.trace {
                                self.dump_state(instr, old_pc);
                            }
                            return Ok(ExitReason::Ecall);
                        } else {
                            return Err(format!("Unsupported ecall code {}", self.cpu.read_reg(17)));
                        }
                    } else if instr == 0x00100073 {
                        // EBREAK - ignored per assignment
                        return Err("Encountered ebreak which is not supported".to_string());
                    } else {
                        return Err("Unsupported system instruction".to_string());
                    }
                }
                _ => return Err(format!("Unknown opcode: 0x{opcode:x}")),
            }

            self.cpu.pc = next_pc;
            self.cpu.regs[0] = 0;

            if self.trace {
                self.dump_state(instr, old_pc);
            }
        }
    }

    fn dump_state(&self, instr: u32, old_pc: u32) {
        println!("pc=0x{old_pc:08x} instr=0x{instr:08x}");
        for (i, value) in self.cpu.regs.iter().enumerate() {
            println!("x{:02}: {:#010x} ({})", i, value, *value as i32);
        }
        println!("-----------------------------");
    }

    fn write_register_dump(&self, path: &PathBuf) -> io::Result<()> {
        let mut file = File::create(path)?;
        for reg in self.cpu.regs.iter() {
            file.write_all(&reg.to_le_bytes())?;
        }
        Ok(())
    }
}

fn decode_i_imm(instr: u32) -> i32 {
    sign_extend((instr >> 20) as u32, 12)
}

fn decode_s_imm(instr: u32) -> i32 {
    let imm11_5 = (instr >> 25) & 0x7f;
    let imm4_0 = (instr >> 7) & 0x1f;
    sign_extend((imm11_5 << 5) | imm4_0, 12)
}

fn decode_b_imm(instr: u32) -> i32 {
    let bit12 = (instr >> 31) & 0x1;
    let bit11 = (instr >> 7) & 0x1;
    let bits10_5 = (instr >> 25) & 0x3f;
    let bits4_1 = (instr >> 8) & 0xf;
    let imm = (bit12 << 12) | (bit11 << 11) | (bits10_5 << 5) | (bits4_1 << 1);
    sign_extend(imm, 13)
}

fn decode_j_imm(instr: u32) -> i32 {
    let bit20 = (instr >> 31) & 0x1;
    let bits10_1 = (instr >> 21) & 0x3ff;
    let bit11 = (instr >> 20) & 0x1;
    let bits19_12 = (instr >> 12) & 0xff;
    let imm = (bit20 << 20) | (bits19_12 << 12) | (bit11 << 11) | (bits10_1 << 1);
    sign_extend(imm, 21)
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

fn parse_args() -> Result<(PathBuf, PathBuf, bool), String> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args.contains(&"--help".to_string()) {
        print_usage();
        return Err("".into());
    }

    let mut trace = false;
    args.retain(|arg| {
        if arg == "--trace" {
            trace = true;
            false
        } else {
            true
        }
    });

    let input = args.get(0).ok_or_else(|| "Input binary not specified".to_string())?;
    let output = if let Some(path) = args.get(1) {
        PathBuf::from(path)
    } else {
        PathBuf::from("regdump.bin")
    };

    Ok((PathBuf::from(input), output, trace))
}

fn print_usage() {
    eprintln!("Usage: riscv-sim <input_bin> [output_bin] [--trace]");
    eprintln!("If output_bin is omitted, registers are written to regdump.bin");
}

fn main() {
    let (input_path, output_path, trace) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("Error: {e}");
            }
            std::process::exit(1);
        }
    };

    let mut program_file = match File::open(&input_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {e}", input_path.display());
            std::process::exit(1);
        }
    };
    let mut program = Vec::new();
    if let Err(e) = program_file.read_to_end(&mut program) {
        eprintln!("Failed to read {}: {e}", input_path.display());
        std::process::exit(1);
    }

    let mut simulator = Simulator::new(program, trace);
    match simulator.run() {
        Ok(reason) => {
            if let Err(e) = simulator.write_register_dump(&output_path) {
                eprintln!("Failed to write register dump to {}: {e}", output_path.display());
                std::process::exit(1);
            }

            println!("Program finished: {:?}", reason);
            for (i, value) in simulator.cpu.regs.iter().enumerate() {
                println!("x{:02} = {}", i, *value as i32);
            }
        }
        Err(e) => {
            eprintln!("Simulation error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn sign_extension() {
        assert_eq!(sign_extend(0b0111_1111_1111, 12), 0x7ff);
        assert_eq!(sign_extend(0b1000_0000_0000, 12), -2048);
    }

    #[test]
    fn decode_b_immediate() {
        // Construct a BEQ with an immediate of -4 to confirm sign handling
        // imm[12|10:5|4:1|11] = 1 111111 1110 1 => -4 when sign-extended
        let instr: u32 = (1 << 31) // imm[12]
            | (0b111111 << 25) // imm[10:5]
            | (0b1110 << 8) // imm[4:1]
            | (1 << 7) // imm[11]
            | 0x63; // opcode with rs/rd/funct3 set to zero for simplicity
        let imm = decode_b_imm(instr);
        assert_eq!(imm, -4);
    }

    fn task_path(dir: &str, name: &str, ext: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests")
            .join(dir)
            .join(format!("{name}.{ext}"))
    }

    fn run_program(bin: &PathBuf) -> [u32; 32] {
        let program = std::fs::read(bin).expect("failed to read test binary");
        let mut simulator = Simulator::new(program, false);
        let exit = simulator.run().expect("simulation failed");
        assert_eq!(exit, ExitReason::Ecall, "program did not terminate via ecall");
        simulator.cpu.regs
    }

    fn load_expected(res: &PathBuf) -> [u32; 32] {
        let data = std::fs::read(res).expect("failed to read reference dump");
        assert_eq!(data.len(), 32 * 4, "unexpected register dump length");

        let mut regs = [0u32; 32];
        for (i, chunk) in data.chunks_exact(4).enumerate() {
            regs[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        regs
    }

    fn check_task(dir: &str, cases: &[&str]) {
        for case in cases {
            let bin = task_path(dir, case, "bin");
            let res = task_path(dir, case, "res");
            let regs = run_program(&bin);
            let expected = load_expected(&res);
            assert_eq!(regs, expected, "Mismatch in {dir}/{case}");
        }
    }

    #[test]
    fn lecturer_task_assembly_programs_match_reference_outputs() {
        check_task("task1", &["addlarge", "addneg", "addpos", "bool", "set", "shift", "shift2"]);
        check_task("task2", &["branchcnt", "branchmany", "branchtrap"]);
        check_task("task3", &["loop", "recursive", "string", "width"]);
        check_task(
            "task4",
            &["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14", "t15"],
        );
    }
}
