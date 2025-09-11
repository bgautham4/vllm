#!/usr/bin/env bash

for d in ./logs/*; do 
    prefill_len=$(basename "$d" |  grep -o '[0-9]\+')
    ./parse_logs.py --log-dir "$d" --prefill-len "$prefill_len" >> "data_${prefill_len}.csv"
    sed -i '1i ntoks,time_ms' "data_${prefill_len}.csv"
    echo "Parsed logs for $prefill_len"
done
