#!/bin/bash

function disp_help {
        echo "Usage: $(basename "$0") [-h|--help] [--model M] [--token-budget T] [--ilen I]"
}

TEMP=$(getopt -o 'h' -l 'help,model:,token-budget:,ilen:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
TB=128
ILEN=1024

while true; do
        case "$1" in
                '-h'|'--help')
                        disp_help
                        exit 0
                ;;
                '--model')
                        MODEL="$2"
                        shift 2
                        continue
                ;;

                '--token-budget')
                        TB="$2"
                        shift 2
                        continue
                ;;

                '--ilen')
                        ILEN="$2"
                        shift 2
                        continue
                ;;

                '--')
                        shift
                        break
                ;;

                *)
                        echo "Invalid argument $1"
                        echo "use -h to display usage"
                        exit 1
                ;;
        esac
done

export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

./run.sh --model "$MODEL" --token-budget "$TB" --max-num-seqs 1 --num-prompts 200 --ilen "$ILEN" --olen 1 -- --profile-scheduler --time-model
mv ./logs/vllm_logs.jsonl ./results/multi_chunked_prefill_"$ILEN"_"$TB".jsonl
