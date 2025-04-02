#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--max-bsize B] [--upto T]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--max-bsize=256"
        echo "--upto=2048, vary input len to upto"
}

function start_server {
        local bsize="$1"
        local input_len="$2"
        vllm serve "$MODEL"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 \
                --max-model-len "$((input_len + 200))" \
                --max_num_seqs "$bsize" \
                --prefill_batch_size 1 \
                --max_num_batched_tokens 4096 \
                --enable_model_timings &
}

function run_benchmark {

        for ((ilen=32;ilen<=UPTO;ilen*=2)); do
                for ((bsize=1;bsize<=MAX_BSIZE;bsize*=2)); do
                        start_server "$bsize" "$ilen"
                        sleep 60 #Sleep to ensure server startup is complete
                        sudo nvidia-smi --lock-gpu-clocks=1380,1380
                        #Run benchmark
                        python benchmark_serving.py --backend vllm \
                                --model "$MODEL" \
                                --max-model-len "$((ilen + 200))" \
                                --dataset-name random \
                                --num_prompts "$bsize" \
                                --random-input-len "$ilen" --random-output-len 100 \
                                --ignore-eos \
                                --experiment-mode BACKLOGGED 

                        #Kill server process
                        kill -SIGTERM "$!"
                        #wait for cleanup
                        sleep 10
                        #Process logs
                        mv logs/vllm_logs.jsonl "results/log_ilen_${ilen}_bsize_${bsize}.jsonl" 
                done
        done
}

TEMP=$(getopt -o 'h' -l 'model:,max-bsize:,upto:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
UPTO="2048"
MAX_BSIZE="256"
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
                '--upto')
                        UPTO="$2"
                        shift 2
                        continue
                ;;
                '--max-bsize')
                        MAX_BSIZE="$2"
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
cd "${0%/*}"
export VLLM_LOGGING_CONFIG_PATH="$(pwd)/log_conf/config.json" 
if [[ ! -d 'logs' ]]; then
        mkdir logs
fi
if [[ ! -d 'results' ]]; then
        mkdir results
fi
run_benchmark 
