#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--ilen input length] [--p-geometric p] [--max-seq-len L] [--max-num-seqs N]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen =100"
        echo "--p--geometric=0.01"
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
                        --dataset-name random \
                        --num_prompts 40000 \
                        --random-input-len "$ILEN" --p-geometric "$PROB" \
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

TEMP=$(getopt -o 'h' -l 'model:,ilen:,p-geometric:,max-num-seqs:,max-seq-len:,token-lim:' -- "$@")
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
echo "Using input prompt length: $ILEN"
echo "Using output prompt length: $PROB"
echo "Using max model len : $MAX_L"
cd "${0%/*}"
cd ../
export VLLM_LOGGING_CONFIG_PATH="$(pwd)/log_conf/config.json" 
if [[ ! -d 'logs' ]]; then
        mkdir logs
fi
if [[ ! -d 'results' ]]; then
        mkdir results
fi
run_benchmark 
