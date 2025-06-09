#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--token-budget T]"
        echo "Defaults:"
        echo "--model=$MODEL"
        echo "--token-budget=$TB"
}

function start_server {
        local bsize="$1"
        local input_len="$2"
        local decode_len="$3"
        local max_model_len=4096
        [ "$max_model_len" -lt $((input_len + decode_len)) ] && ((max_model_len = input_len + decode_len))
        echo "Max model len: $max_model_len"
        VLLM_USE_V1=1 vllm serve "$MODEL" \
                --port 8000 \
                --max-model-len "$max_model_len" \
                --max_num_seqs "$bsize" \
                --max_num_batched_tokens "$TB" \
                --profile-scheduler \
                --profile-model &
}

function run_benchmark {
        for ((dp=0;dp<=100;dp+=10)); do
                #Parameter calculations
                ((max_num_seqs = (TB*dp)/100))
                [ "$max_num_seqs" -lt 1 ] && max_num_seqs=1
                ((output_len = max_num_seqs + 1))
                ((input_len = TB - max_num_seqs))
                [ "$input_len" -lt 1 ] && input_len=1

                echo "Using batch size = $max_num_seqs"
                echo "Using input len = $input_len"
                echo "Using decode len = $output_len"

                start_server "$bsize" "$input_len" "$output_len"
                sleep 60 #Sleep to ensure server startup is complete
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --dataset-name random \
                        --num_prompts 1000 \
                        --random-input-len "$input_len" --random-output-len "$output_len" \
                        --ignore-eos

                #Kill server process
                kill -SIGTERM "$!"
                #wait for process shutdown
                sleep 10
        done
}

function reset_clocks {
        sudo nvidia-smi --reset-gpu-clocks
        sudo nvidia-smi --reset-memory-clocks
        sudo nvidia-smi --persistence-mode=0
}

function cleanup {
        reset_clocks
        pkill vllm
}

TEMP=$(getopt -o 'h' -l 'model:,token-budget:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
TB="256"
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
                '--token-budget')
                        TB="$2"
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

echo "Using model: $MODEL"
echo "Using token budget of $TB"
cd "${0%/*}"
if [[ ! -d 'results' ]]; then
        mkdir results
fi

trap "echo 'SIGTERM received!....stopping running vLLM instances'; cleanup; exit 1" SIGTERM SIGINT

#Lock gpu clocks
sudo nvidia-smi --persistence-mode=1
#Change this frequency to base clock
sudo nvidia-smi --lock-gpu-clocks=1380,1380
#Change this to fastest supported memory clocks for the given base clock
sudo nvidia-smi --lock-memory-clocks=

#configure logging
export VLLM_LOGGING_CONFIG_PATH=./configs/logger.json

run_benchmark

reset_clocks
