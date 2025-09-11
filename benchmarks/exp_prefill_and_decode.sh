#!/bin/bash

function disp_help {
        echo "Usage: $(basename "$0") model" 
}

[ "$#" -ne 1 ] && disp_help && exit 1
[ "$1" = "-h" ] || [ "$1" = "--help" ] && disp_help && exit 0

#configure logging
export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

MODEL="$1"

TB=256 #Adjust as needed

cd "${0%/*}" || exit 1

trap 'pkill exp.sh; exit 1' SIGTERM SIGINT

for ((ilen=8; ilen<=TB/2; ilen*=2)); do
        res_dir="./results/logs_npt_${ilen}"
        if [ -d "$res_dir" ]; then 
                rm "$res_dir"/*
        else
                mkdir "$res_dir"
        fi
        for ((i=1; i<TB-ilen; i+=5)); do
                ./run.sh --model "$MODEL" --token-budget "$TB" --max-num-seqs "$((i + 1))" --num-prompts 5000 --ilen "$ilen" --olen "$((i + 10))" -- --profile-scheduler --time-model
                mv logs/vllm_logs.jsonl "${res_dir}/log_${i}.jsonl"
        done
done
