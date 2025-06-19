#!/bin/bash

#Display help text
function disp_help {
        echo "Usage: $(basename "$0") [-h|--help] [--model model] [--token-budget T] [--max-num-seqs S] [--num-prompts N] [--ilen I] [--olen O] [-- --addional-arguments]"
        echo "Defaults:"
        echo "--model=$MODEL"
        echo "--token-budget=$TB"
        echo "--max-num-seqs=$MAX_BSIZE"
        echo "--num-prompts=$NPROMPTS"
        echo "--ilen=$ILEN"
        echo "--olen=$OLEN"
        echo "Additional arguments after -- will be passed as arguments to vllm"
        echo "See vllm serve --help to get available list of arguments for vllm"
}

function start_server {
        local max_model_len=4096
        [ "$max_model_len" -lt $((ILEN + OLEN)) ] && ((max_model_len = ILEN + OLEN))
        echo "Max model len: $max_model_len"
        VLLM_USE_V1=1 vllm serve "$MODEL" \
                --port 8000 \
                --max-model-len "$max_model_len" \
                --max_num_seqs "$MAX_BSIZE" \
                --max_num_batched_tokens "$TB" \
                --max-num-partial-prefills 1 \
                "${VLLM_OPTS[@]}" &
}

function run_benchmark {
        start_server 

        num_retries=200
        while ! curl -sf http://localhost:8000/health > /dev/null; do
                ((--num_retries))
                if [ "$num_retries" -lt 0 ]; then
                        echo "Server startup timed out. Exiting...."
                        kill -SIGTERM "$BASHPID"
                fi
                sleep 1
        done

        #Run benchmark
        python benchmark_serving.py --backend vllm \
                --model "$MODEL" \
                --dataset-name random \
                --num_prompts "$NPROMPTS" \
                --random-input-len "$ILEN" --random-output-len "$OLEN" \
                --ignore-eos

        #Kill server process and wait for shutdown
        kill -SIGTERM "$!"
        sleep 10
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

TEMP=$(getopt -o 'h' -l 'help,model:,token-budget:,max-num-seqs:,num-prompts:,ilen:,olen:' -- "$@")
if [[ $? -ne 0 ]];then
        echo 'getopt error, Terminating...' >&2
        echo 'Use -h to display help text.'
        exit 1
fi
eval set -- "$TEMP"
unset TEMP

MODEL="facebook/opt-350m"
TB=256
MAX_BSIZE=64
NPROMPTS=1000
ILEN=256
OLEN=128
declare -a VLLM_OPTS

while true; do
        case "$1" in
                '-h'|'--help')
                        disp_help
                        exit 0
                ;;
                '--model')
                        MODEL="$2"
                        shift 2
                ;;

                '--token-budget')
                        TB="$2"
                        shift 2
                ;;
                '--max-num-seqs')
                        MAX_BSIZE="$2"
                        shift 2
                ;;

                '--num-prompts')
                        NPROMPTS="$2"
                        shift 2
                ;;

                '--ilen')
                        ILEN="$2"
                        shift 2
                ;;

                '--olen')
                        OLEN="$2"
                        shift 2
                ;;

                '--')
                        shift
                        break
                ;;

        esac
done

VLLM_OPTS+=("$@")

echo "Using model: $MODEL"
echo "Using token budget of $TB"
echo "Using max batch size of $MAX_BSIZE"
echo "Using $NPROMPTS prompts"
echo "Using input length of $ILEN"
echo "Using output length of $OLEN"

cd "${0%/*}"
if [[ ! -d 'results' ]]; then
        mkdir results
fi

trap "echo 'SIGTERM received!....stopping running vLLM instances'; cleanup; exit 1" SIGTERM SIGINT

#Lock gpu clocks
sudo nvidia-smi --persistence-mode=1
#Change this frequency to base clock
sudo nvidia-smi --lock-gpu-clocks=2205,2205
#Change this to fastest supported memory clocks for the given base clock
sudo nvidia-smi --lock-memory-clocks=11201,11201

run_benchmark

reset_clocks
