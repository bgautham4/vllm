#!/bin/bash

#Display help text
function disp_help {
        echo "Help text"
        echo
        echo "Usage: [-h] [--model model] [--ilen input length] [--olen output length] [--log_dir log directory]"
        echo "Defaults:"
        echo "--model=facebook/opt-350m"
        echo "--ilen == --olen = 100"
        echo "--log_dir=None, will be created automatically."
}

function start_server {
        local mdl="$1"
        local bsize="$2"
        local il="$3"
        local ldir="$4"
        local token_lim=$(($il * $bsize))
        if [[ "$token_lim" -lt 2048 ]];then
                token_lim=2048
        fi

        if [[ "$token_lim" -gt 32768 ]];then
                token_lim=32768
        fi

        echo "Token budget set to $token_lim"

        vllm serve "$mdl"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 --batched-mode \
                --max_num_seqs "$bsize" \
                --prefill_batch_size "$bsize" \
                --max_num_batched_tokens "$token_lim" \
                --mqllm_ec_log_dir "$ldir" &
}

function run_benchmark {
        local mdl="$1"
        local il="$2"
        local ol="$3"
        local ldir="$4"

        if [[ -z "$ldir" ]];then
                mkdir /tmp/vllm_temp_log_dir/ 
                ldir='/tmp/vllm_temp_log_dir/'
        fi

        echo "Logs written to: $ldir"

        for ((i=1;i<100;i=i+5)); do
                start_server "$mdl" "$i" "$il" "$ldir"
                sleep 40 #Sleep to ensure server startup is complete
                sudo nvidia-smi --lock-gpu-clocks=1410,1410
                sudo nvidia-smi --lock-memory-clocks=5001,5001
                #Run benchmark
                python benchmark_serving.py --backend vllm \
                        --model "$mdl" \
                        --dataset-name random \
                        --num_prompts $((i*100)) \
                        --random-input-len "$il" --random-output-len "$ol" \
                        --experiment-mode BATCHED --batch_size "$i"

                res=$(cat "$ldir"/*.txt | awk 'BEGIN{x = 0;y = 0;}{x += $1;y += $2;}END{printf("%f %f",x/NR,y/NR);}')
                if [[ ! -d 'results ']];then
                        mkdir results
                fi
                echo "$i $res" >> results/results_batching.txt
                rm "$log_dir"/*.txt
                #Kill server process
                kill -SIGTERM "$!"
        done
}

TEMP=$(getopt -o 'h' -l 'model:,ilen:,olen:,log-dir:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        disp_help
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

model="facebook/opt-350m"
ilen=100
olen=100
log_dir=
while true; do
        case "$1" in
                '-h')
                        disp_help
                        exit 0
                ;;
                '--model')
                        model="$2"
                        shift 2
                        continue
                ;;
                '--ilen')
                        ilen="$2"
                        shift 2
                        continue
                ;;
                '--olen')
                        olen="$2"
                        shift 2
                        continue
                ;;
                '--log-dir')
                        log_dir="$2"
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

echo "Using model: $model"
echo "Using input prompt length: $ilen"
echo "Using output prompt length: $olen"
run_benchmark "$model" "$ilen" "$olen" "$log_dir"
