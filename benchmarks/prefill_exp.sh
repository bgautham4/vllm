#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--max-bsize T]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen=$ILEN" 
        echo "--max-bsize=$MAX_BSIZE"
}

function start_server {
        local bsize="$1"
        local input_len="$2"
        local token_lim=$((2 * bsize * input_len))

        vllm serve "$MODEL"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 \
                --max-model-len "$((input_len + 100))" \
                --max_num_seqs "$bsize" \
                --prefill_batch_size "$bsize" \
                --enable-model-timings \
                --max_num_batched_tokens "$token_lim" & 
}

function run_benchmark {
        for ((bsize=1;bsize<=MAX_BSIZE;bsize+=2)); do # Run experiments max-bsize $MAX_BSIZE input tokens total
                start_server "$bsize" "$ILEN"
                sleep 100 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1380,1380
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --dataset-name random \
                        --num_prompts $((bsize*100)) \
                        --random-input-len "$ILEN" --random-output-len 1 \
                        --ignore-eos

                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 10
                #Process logs
                mkdir -p results/logs_"$bsize"
                mv trace_* "results/logs_${bsize}/"
        done
}

TEMP=$(getopt -o 'h' -l 'model:,max-bsize:,ilen:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
MAX_BSIZE='256'
ILEN='32'
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
                '--max-bsize')
                        MAX_BSIZE="$2"
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
                        echo "$1"
                        echo "Invalid arguments"
                        echo "use -h to display usage"
                        exit 1
                ;;
        esac
done

echo "Using model: $MODEL"
cd "${0%/*}"
if [[ ! -d 'results' ]]; then
        mkdir results
fi
run_benchmark 
