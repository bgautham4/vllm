#!/usr/bin/env bash

for f in ./decode_logs/*; do 
    bs=$(basename "$f" |  grep -o '[0-9]\+')
    ./parse_logs.py --bs "$bs" --log-file "$f" >> tmp.csv
    echo "Parsed log file $f"
done
sort -n tmp.csv > parsed_data.csv && rm tmp.csv
sed -i '1i ntoks,time_ms,throughput' parsed_data.csv
