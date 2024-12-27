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
                --port 8000 \
                --max_num_seqs "$bsize" \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" \
                --mqllm_ec_log_dir "$LOG_DIR" &
}

function run_benchmark {
        echo "Using $LOG_DIR as temp log directory"

        for ((i=1;i<100;i=i+5)); do
                start_server "$i"
                sleep 80 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1410,1410
                sudo nvidia-smi --lock-memory-clocks=5001,5001
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --dataset-name random \
                        --num_prompts $(( $i * 100 )) \
                        --random-input-len "$ILEN" --random-output-len "$OLEN" \
                        --experiment-mode BACKLOGGED 

                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 20
                #Process logs
                res=$(awk -v bsize="$i" -v ilen="$ILEN" '
                BEGIN{tbt_mean=0;tbt_std=0;pft_mean=0;pft_std=0;}
                /decode:.*/{tbt_mean=$2;tbt_std=$3;}
                /ttft:.*/{pft_mean=$2;pft_std=$3;}
                END{printf("(%f,%f) (%f,%f)", pft_mean, pft_std, tbt_mean, tbt_std);}
                ' "$LOG_DIR/metrics.txt")
                if [[ ! -d 'results' ]];then
                        mkdir results
                fi 
                echo "$i $res" >> results/results_batching.txt
                rm "$LOG_DIR"/*.txt
        done
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
LOG_DIR="/tmp/$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 13)"
mkdir -p "$LOG_DIR" || exit
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
