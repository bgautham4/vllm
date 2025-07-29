#!/usr/bin/env bash

# Assumes directory is named as <some_string>_<batch size as a number>
# Each directory contains many traces, using 81st trace as it is just after the
# test profiling run

for d in traces/*; do
        if [ ! -d "$d" ]; then
                continue
        fi
        bs=$(basename "$d" | grep -o '[0-9]\+')
        f="$d/trace_81.json"
        dat=$(jq --compact-output -r '[.traceEvents | .[] | select(.name | test("ampere|cutlass";"i"))][1] | .name,.dur,.args.grid,.args.block' "$f")
        printf '"%d",' "$bs"
        while read -r line; do
                printf '"%s",' "$line"
        done <<< "$dat"
        printf "\n"
done
