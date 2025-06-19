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
TB=1024
ILEN=128

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
                        continue
                ;;

                *)
                        echo "Invalid argument $1"
                        echo "use -h to display usage"
                        exit 1
                ;;
        esac
done

export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

trap 'pkill exp.sh; exit 1' SIGTERM SIGINT
for ((i=1; i<TB-ILEN; i+=5)); do
        ./run.sh --model "$MODEL" --token-budget "$TB" --max-num-seqs "$((i + 1))" --num-prompts 1000 --ilen "$ILEN" --olen "$((i + 10))" -- --profile-scheduler --time-model
        mv ./logs/vllm_logs.jsonl ./results/log_"$ILEN"_"$TB".jsonl
done
