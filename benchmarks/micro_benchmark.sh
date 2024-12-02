#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--ilen input length] [--olen output length]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen == --olen = 100"
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
                --port 8000 --batched_mode \
                --max_num_seqs "$bsize" \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" \
                --model-stats-log-dir "$LOG_DIR" &
}

function run_benchmark {
        echo "Logs written to: $LOG_DIR"

        for ((i=1;i<100;i*=2)); do
                start_server "$i"
                sleep 80 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1410,1410 #Change GPU mem and SM frequencies here
                sudo nvidia-smi --lock-memory-clocks=5001,5001
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --dataset-name random \
                        --num_prompts "$i" \
                        --random-input-len "$ILEN" --random-output-len "$OLEN" \
                        --experiment-mode BATCHED --batch_size "$i"

                mv "$LOG_DIR"/temp.txt temp.txt
                if [[ ! -f "temp_old.txt" ]];then
                        mv temp.txt res.txt
                else
                        paste -d',' temp_old.txt temp.txt > res.txt
                fi
                mv res.txt temp_old.txt
                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 10
        done
        rm temp.txt
        if [[ ! -d 'results' ]];then
                mkdir results
        fi
        mv temp_old.txt results/mb_res.txt
}

TEMP=$(getopt -o 'h' -l 'model:,ilen:,olen:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
ILEN=100
OLEN=100
LOG_DIR="/tmp/$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 13)" #Temp dir for temp logs
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
                '--olen')
                        OLEN="$2"
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
echo "Using output prompt length: $OLEN"
run_benchmark 
rm -rf "$LOG_DIR"
