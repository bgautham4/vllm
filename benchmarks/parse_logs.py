import json
import os
import glob
import argparse
from typing import List, Dict


def main(args):
    log_files = glob.glob(os.path.join(args.log_directory, '*.jsonl'))
    prefill_samples: Dict[int, List[float]] = {}
    decode_samples: Dict[int, List[float]] = {}
    for log_file in log_files:
        processed_lines = []
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                event_type = entry.get('message')
                assert event_type is not None
                if event_type not in {'SCHED_START', 'SCHED_END', 'MODEL_EXEC_END'}:
                    continue
                processed_lines.append(entry)
        i = 0
        while (i + 2 < len(processed_lines)):
            sched_start = processed_lines[i]
            if (sched_start.get('message') != 'SCHED_START'):
                i += 1
                continue
            sched_end = processed_lines[i+1]
            proc_end = processed_lines[i+2]
            i += 3
            assert sched_end.get('message') == 'SCHED_END'
            assert proc_end.get('message') == 'MODEL_EXEC_END'
            task: str = sched_end.get('TASK')
            batch_size: int = len(sched_end.get('ids'))
            time_taken: float = proc_end.get(
                'perf_timer') - sched_start.get('perf_timer')
            if task == 'PREFILL':
                if batch_size in prefill_samples.keys():
                    prefill_samples[batch_size].append(time_taken)
                else:
                    prefill_samples[batch_size] = [time_taken,]
            else:
                if batch_size in decode_samples.keys():
                    decode_samples[batch_size].append(time_taken)
                else:
                    decode_samples[batch_size] = [time_taken,]
        with open('out.json', 'w') as f:
            d = {"prefill_data": prefill_samples,
                 "decode_data": decode_samples}
            json.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse result jsonl files')
    parser.add_argument(
        '--log_directory',
        type=str,
        default=None,
        help='Directory containing logs'
    )
    args = parser.parse_args()
    main(args)
