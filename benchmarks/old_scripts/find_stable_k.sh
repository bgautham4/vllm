#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: [-h] [--model model] [--ilen input length] [--olen-mean output length]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen == 100"
        echo "--olen-mean == 100"
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
                --max_num_seqs 100 \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" &
}

function binary_search_k {
        local X=1 # WARN:Assuming this point is always stable
        local Y=100
        local Y_OLD=100
        for ((i=1;i<100;i=i+1)); do
                X=1
                Y=100
                Y_OLD=100
                start_server "$i"
                sleep 60 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1410,1410
                sudo nvidia-smi --lock-memory-clocks=5001,5001
                #To find k
                while [[ $((Y-X)) -gt 1 ]]; do
                        echo "using rate = $Y requests/sec"
                        python benchmark_serving.py --backend vllm \
                                --model "$MODEL" \
                                --dataset-name random \
                                --num-prompts 1000 \
                                --request-rate "$Y" \
                                --random-input-len "$ILEN" --p-geometric "$PROB" \
                                --experiment-mode REGULAR \
                                --save-result
                        local RES=$(python test_stability.py --rate "$Y" --file metrics.json)
                        if [[ "$RES" == "STABLE" ]];then
                                X="$Y"
                                Y=$((Y_OLD-1))
                        else
                                Y_OLD="$Y"
                                Y=$(( (X+Y) / 2 ))
                        fi
                        rm metrics.json
                done
                #Kill server process
                echo "$i $X $Y" >> results/stability.txt
                kill -SIGTERM "$!"
                sleep 10
        done
}

TEMP=$(getopt -o 'h' -l 'model:,ilen:,olen-mean:' -- "$@")
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
PROB=$(awk '{print 1/$1}' <<< "$OLEN")
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
                '--olen-mean')
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
echo "Using mean output tokens generated: $OLEN"
run_benchmark 
