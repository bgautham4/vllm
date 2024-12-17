#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--ilen input length] [--p-geometric p] [--max-num-seqs N]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen =100"
        echo "--p--geometric=0.01"
        echo "--max-num-seqs=100"
}

function start_server {
        local bsize="$1"
        local token_lim=$(( ($ILEN * 2) * $bsize))
        if [[ "$token_lim" -lt 2048 ]];then
                token_lim=2048
        fi

        if [[ "$token_lim" -gt 32768 ]];then
                token_lim=32768
        fi

        echo "Token budget set to $token_lim"

        vllm serve "$MODEL"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 \
                --max_num_seqs "$MAX_N" \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" &
}

function run_benchmark {

        for ((i=1;i<100;i=i+1)); do
                start_server "$i"
                sleep 80 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1410,1410
                sudo nvidia-smi --lock-memory-clocks=5001,5001
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --dataset-name random \
                        --num_prompts 10000 \
                        --random-input-len "$ILEN" --p-geometric "$PROB" \
                        --experiment-mode BACKLOGGED 

                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 20
                #Process logs
        done
}

TEMP=$(getopt -o 'h' -l 'model:,ilen:,p-geometric:,max-num-seqs:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
ILEN='100'
PROB='0.01'
MAX_N='100'
while true; do
        case "$1" in
                '-h')
                        disp_help
                        exit 0
                ;;
                '--model')
                        MODEL="$2"
                        shift 2
                        continue
                ;;
                '--ilen')
                        ILEN="$2"
                        shift 2
                        continue
                ;;
                '--p-geometric')
                        PROB="$2"
                        shift 2
                        continue
                ;;
                '--max-num-seqs')
                        MAX_N="$2"
                        shift 2
                        continue
                        ;;
                '--')
                        shift
                        break
                ;;
                *)
                        echo "$1"
                        echo "Invalid arguments"
                        echo "use -h to display usage"
                        exit 1
                ;;
        esac
done

echo "Using model: $MODEL"
echo "Using input prompt length: $ILEN"
echo "Using output prompt length: $PROB"
cd "${0%/*}"
export VLLM_LOGGING_CONFIG_PATH="$(pwd)/log_conf/config.json" 
run_benchmark 
rm -rf "$LOG_DIR"
