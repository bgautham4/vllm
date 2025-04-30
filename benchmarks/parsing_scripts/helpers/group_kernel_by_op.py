#!/usr/bin/env python3

import csv
import os
import re
from collections import defaultdict
from sys import argv

# Define the ops and their associated regex
ops = ["kqv_proj", "rope", "paged_attention", "o_proj", "gated_up_proj", "down_proj"]
op_re = [
    r"(cutlass)|(gemv)|(gemm)",
    r"rotary",
    r"attention",
    r"(cutlass)|(gemv)|(gemm)",
    r"(cutlass)|(gemv)|(gemm)",
    r"(cutlass)|(gemv)|(gemm)",
]
N_OPS = len(ops)

# Function to strip quotes
def strip_quotes(s):
    return s[1:-1] if s.startswith('"') and s.endswith('"') else s

# Function to split dimension strings
def split_block_and_grid_dims(dimstr):
    dimstr = dimstr.strip("[]")
    return [d.strip() for d in dimstr.split(",")]

# Simulated list of input files for processing (example)
input_files = argv[1:] 
# Output collection
output_rows = []

# Process each file
for filepath in input_files:
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        
        match_obj = re.search(r"[0-9]+", os.path.basename(filepath))
        bsize = match_obj.group(0) if match_obj else "unknown"
        
        op_idx = 0
        kernel_info_table = set()
        
        for row in reader:
            if op_idx >= N_OPS or len(row) < 9:
                continue

            kernel = strip_quotes(row[0])
            gdim_str = row[1]
            bdim_str = row[2]

            if not gdim_str or not bdim_str:
                continue

            gdim = split_block_and_grid_dims(strip_quotes(gdim_str))
            bdim = split_block_and_grid_dims(strip_quotes(bdim_str))

            op_type = "unknown"
            if re.search(op_re[op_idx], kernel):
                op_type = ops[op_idx]
                op_idx += 1

            entry = (kernel, *gdim, *bdim, row[5], row[6], row[7], row[8], op_type)
            kernel_info_table.add(entry)
        
        for entry in kernel_info_table:
            output_rows.append([bsize] + list(entry))

# Write to CSV
output_path = "kernel_dat.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "bsize", "kernel", "gdim_x", "gdim_y", "gdim_z",
        "bdim_x", "bdim_y", "bdim_z", "blocks_per_sm",
        "warps_per_sm", "regs_per_thread", "occupancy", "op_type"
    ])
    writer.writerows(output_rows)
