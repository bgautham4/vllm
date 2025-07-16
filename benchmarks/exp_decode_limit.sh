#!/bin/bash


function disp_help {
        echo "Usage: $(basename "$0") model" 
}

[ "$#" -ne 1 ] && disp_help && exit 1
[ "$1" = "-h" ] || [ "$1" = "--help" ] && disp_help && exit 0

cd "${0%/*}" || exit 1
#configure logging
export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

MODEL="$1"

trap 'pkill run.sh; exit 1' SIGINT SIGTERM

for ((i=4;i<2048;i*=2)); do
        ./run.sh --model "$MODEL" --token-budget "$((32*i))" --max-num-seqs "$i" --num-prompts 1000 --ilen "$((2048/i))" --olen 32 -- --profile-scheduler --time-model
        mv logs/vllm_logs.jsonl "./results/log_bs_${i}.jsonl"
done
