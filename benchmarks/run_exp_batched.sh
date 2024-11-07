#!/bin/bash

#Arguments : model prompt_input_len log_dir
model="$1"
input_len="$2"
log_dir="$3"

function start_server {
        local token_lim=$(($1*$input_len))

        if [[ "$token_lim" -lt 2048 ]];then
                token_lim=2048
        fi

        if [[ "$token_lim" -gt 32768 ]];theb
                token_lim=32768
        fi

        echo "Token budget set to $token_lim"

        vllm serve "$model"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 --batched-mode \
                --max_num_seqs "$1" \
                --prefill_batch_size "$1" \
                --max_num_batched_tokens "$token_lim" \
                --mqllm_ec_log_dir "$log_dir" &
}

for ((i=1;i<100;i=i+5)); do
        start_server "$i"
        sleep 40 #Sleep to ensure server startup is complete
        sudo nvidia-smi --lock-gpu-clocks=1410,1410
        sudo nvidia-smi --lock-memory-clocks=5001,5001
        #Run benchmark
        python benchmark_serving.py --backend vllm --model "$model" --dataset-name random --num_prompts $((i*1000)) --random-input-len "$input_len" --random-output-len 100 --experiment-mode BATCHED --batch_size "$i"

        res=$(cat "$log_dir"/*.txt | awk 'BEGIN{x = 0;y = 0;}{x += $1;y += $2;}END{printf("%f %f",x/NR,y/NR);}')
        echo "$i $res" >> results/results_batching.txt
        rm "$log_dir"/*.txt
        #Kill server process
        kill -SIGTERM "$!"
done

