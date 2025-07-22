#!/bin/bash

cd "${0%/*}" 

if [[ ! -d '../results' ]]; then
        echo "No results to parse....Exiting."
        exit 1
fi

mkdir -p parsed_csvs

for d in ../results/logs_*; do
        bsize=$(basename "$d" | cut -d'_' -f2)
        ./helpers/parse_json_trace_full.py "${d}/trace_50.json"
        mv out.csv "parsed_csvs/data_${bsize}.csv"
done

./helpers/get_op_time.awk parsed_csvs/* | column -t > op_times.tsv
./helpers/get_op_time_detailed.awk parsed_csvs/* | column -t > op_times_detailed.tsv
./helpers/group_kernel_by_op.py parsed_csvs/*
