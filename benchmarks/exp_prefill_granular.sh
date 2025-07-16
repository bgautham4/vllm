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

declare -a TOKS
TOKS+=($(seq -s' ' 2 2 38)) #start step stop
TOKS+=($(seq -s' ' 40 5 95))
TOKS+=($(seq -s' ' 100 10 190))
TOKS+=($(seq -s' ' 200 20 520))
TOKS+=($(seq -s' ' 540 20 1040))
TOKS+=($(seq -s' ' 1050 50 2050))
#Add more points if needed
cd "${0%/*}"
for i in "${TOKS[@]}"; do
        ./run.sh --model "$MODEL" --token-budget "$i" --max-num-seqs 1 --num-prompts 200 --ilen "$i" --olen 1 -- --profile-scheduler --time-model
        mv logs/vllm_logs.jsonl "./results/log_tb_${i}.jsonl"
done
