#!/usr/bin/env python

import json
import numpy as np
import argparse

def main(args):
    bs = args.bs
    assert bs is not None
    log_file = args.log_file 
    assert log_file is not None
    with open(log_file, "r") as f:
        lines = f.readlines()
        i = 0
        exec_times = {
            "ATTN": [],
            "MLP": []
        }
        while i < len(lines) - 1:
            sched_data = json.loads(lines[i])
            i += 1
            if sched_data['message'] != "SCHEDULER":
                continue
            any_prefills = filter(lambda x:x>1, sched_data['scheduler_output'].values())
            if len(set(any_prefills)) > 0:
                continue
            bs_curr = sum(sched_data['scheduler_output'].values())
            if bs_curr < bs:
                continue
            times = {
                "ATTN": [],
                "MLP": [],
            }
            while True:
                model_data = json.loads(lines[i])
                if model_data['message'] != "CUDA_TIMER":
                    break
                i += 1
                op = model_data['op']
                exec_time = model_data['time_taken_ms']
                times[op].append(exec_time) 

            for op, time_list in times.items():
                exec_times[op].append(np.mean(time_list))
    print(f'{bs},{np.mean(exec_times["ATTN"])},{np.mean(exec_times["MLP"])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="parse.py",
                                     description="parse log files to get execution time")
    parser.add_argument(
        '--bs',
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
