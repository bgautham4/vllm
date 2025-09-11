#!/usr/bin/env python

import json
import numpy as np
import argparse
import glob

def main(args):
    prefill_len = args.prefill_len
    assert prefill_len is not None
    log_dir = args.log_dir 
    assert log_dir is not None
    exec_times_dbs = {} 
    for log_file in glob.glob(f'{log_dir}/*.jsonl'): 
        with open(log_file, "r") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines) - 1:
                sched_data = json.loads(lines[i])
                if sched_data['message'] != "SCHEDULER":
                    i += 1
                    continue
                model_data = json.loads(lines[i+1])
                i += 2
                if (model_data['message'] != "MODEL_EXEC"):
                    continue
                bs_exsists = list(
                        filter(lambda x:x>=prefill_len, sched_data['scheduler_output'].values())
                )
                if len(bs_exsists) != 1:
                    continue
                decode_bs = sum(
                    filter(lambda x:x==1, sched_data['scheduler_output'].values())
                )
                exec_time = model_data['time_taken_ms']
                if decode_bs in exec_times_dbs.keys():
                    exec_times_dbs[decode_bs].append(exec_time)
                else:
                    exec_times_dbs[decode_bs] = [exec_time,]

    bsl = sorted(exec_times_dbs.keys())
    for bs in bsl:
        times = exec_times_dbs[bs]
        if len(times) < 10: # Atleast these many samples present?
            continue
        print(f'{bs},{np.mean(times)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="parse.py",
                                     description="parse log files to get execution time")
    parser.add_argument(
        '--prefill-len',
        type=int,
        default=None,
        help="Token budget"
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
