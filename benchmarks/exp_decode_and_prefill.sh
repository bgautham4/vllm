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

cd "${0%/*}"
TB=256 #Adjust as needed
for ((dbs=8; dbs<=TB/2; dbs*=2)); do
        for ((ilen=TB-dbs; ilen>=1; ilen-=5)); do
                ./run.sh --model "$MODEL" --token-budget "$TB" --max-num-seqs "$((dbs+1))" --num-prompts 5000 --ilen "$ilen" --olen "$((dbs+10))" -- --profile-scheduler --time-model
                mv logs/vllm_logs.jsonl "./results/log_tb_${TB}_dbs_${dbs}_npt_${ilen}.jsonl"
        done
done


