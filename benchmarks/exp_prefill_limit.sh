#!/bin/bash

function disp_help {
        echo "Usage: $(basename "$0") model" 
}

[ "$#" -ne 1 ] && disp_help && exit 1
[ "$1" = "-h" ] || [ "$1" = "--help" ] && disp_help && exit 0

#configure logging
export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

MODEL="$1"

trap 'pkill run.sh; exit 1' SIGINT SIGTERM

for ((i=4;i<2048;i*=2)); do
        ./run.sh --model "$MODEL" --token-budget "$i" --max-num-seqs 1 --num-prompts 1000 --ilen "$i" --olen 1 -- --profile-scheduler --time-model
        mv logs/vllm_logs.jsonl "./results/log_tb_${i}.jsonl"
done


