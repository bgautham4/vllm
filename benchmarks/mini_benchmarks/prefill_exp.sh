#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--ilen input length] [--max-seq-len L] [--upto T]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen =$ILEN"
        echo "--max-seq-len=$MAX_L"
        echo "--token-lim=max-seq-len=$TOKEN_LIM -> Adjust token budget"
        echo "--upto=$UPTO tokens -> Adjust the max number(start with ilen and double each time upto T)"
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
                --max_num_seqs "$bsize" \
                --prefill_batch_size "$bsize" \
                --batched-mode \
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

        for ((i=1;i<=(UPTO/ILEN);i*=2)); do # Run experiments upto $UPTO input tokens
                start_server "$i"
                sleep 60 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1380,1380
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$MODEL" \
                        --max-model-len "$MAX_L" \
                        --dataset-name random \
                        --num_prompts $((i*100)) \
                        --random-input-len "$ILEN" --random-output-len 10 \
                        --ignore-eos \
                        --experiment-mode BACKLOGGED 

                #Kill server process
                kill -SIGTERM "$!"
                #wait for cleanup
                sleep 10
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

TEMP=$(getopt -o 'h' -l 'model:,ilen:,max-seq-len:,token-lim:,upto:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
ILEN='32'
MAX_L='2048'
TOKEN_LIM="$MAX_L"
UPTO='8192'
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
                '--upto')
                        UPTO="$2"
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
