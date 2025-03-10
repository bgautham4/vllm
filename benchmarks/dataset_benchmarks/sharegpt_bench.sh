#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--max-seq-len L] [--max-num-seqs N] [--token-lim T]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--max-num-seqs=100 -> Adjust N"
        echo "--max-seq-len=2048"
        echo "--token-lim=max-seq-len=2048 -> Adjust token budget"
}

function start_server {
        local bsize="$1"
        local token_lim="$TOKEN_LIM"
        if [[ "$token_lim" -lt "$MAX_L" ]]; then
                token_lim="$MAX_L"
        fi
        echo "Token budget set to $token_lim"

        vllm serve "$MODEL"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 \
                --max-model-len "$MAX_L" \
                --max_num_seqs "$MAX_N" \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" \
                --mqllm_ec_log_dir ./ &
}

function run_benchmark {
        if [[ -f results/metrics.txt ]]; then
                mv results/metrics.txt results/metrics.txt.bak
        else
                touch results/metrics.txt
                echo "k tpt cmpl_time" >> results/metrics.txt
        fi

        for ((i=1;i<MAX_N;i=i+1)); do
                start_server "$i"
                sleep 80 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=870,870
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --max-model-len "$MAX_L" \
                        --dataset-name sharegpt \
                        --dataset-path ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json\
                        --num_prompts 40000 \
                        --ignore-eos \
                        --experiment-mode BACKLOGGED 

                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 20
                #Process logs
                mv logs/vllm_logs.jsonl "results/log_$i.jsonl" 
                res=$(awk '
                /^tpt:.*/{tpt=$2}
                /^cmpl_time_noq:.*/{cmpl_time=$2}
                END{print tpt,cmpl_time}' ./metrics.txt)
                echo "$i $res" >> results/metrics.txt
                rm ./metrics.txt
        done
}

TEMP=$(getopt -o 'h' -l 'model:,max-num-seqs:,max-seq-len:,token-lim:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
MAX_N='100'
MAX_L='2048'
TOKEN_LIM="$MAX_L"
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
                '--max-num-seqs')
                        MAX_N="$2"
                        shift 2
                        continue
                        ;;
                '--max-seq-len')
                        MAX_L="$2"
                        shift 2
                        continue
                        ;;
                '--token-lim')
                        TOKEN_LIM="$2"
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
echo "Using max model len : $MAX_L"
echo "Using token budget: $TOKEN_LIM"
cd "${0%/*}"
cd ../
export VLLM_LOGGING_CONFIG_PATH="$(pwd)/log_conf/config.json" 
if [[ ! -d 'logs' ]]; then
        mkdir logs
fi
if [[ ! -d 'results' ]]; then
        mkdir results
fi
if [[ ! -f 'datasets/ShareGPT_V3_unfiltered_cleaned_split.json' ]]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
        mv ShareGPT_V3_unfiltered_cleaned_split.json ./datasets
fi
run_benchmark 
