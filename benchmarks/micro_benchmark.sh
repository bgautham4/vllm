#!/bin/bash

#Arguments : model prompt_input_len log_dir
model="$1"
input_len="$1"
log_dir="$2"

function start_server {
        local token_lim=$(($1*$input_len))

        if [[ "$token_lim" -lt 2048 ]];then
                token_lim=2048
        else
                token_lim=32768
        fi

        echo "Token limit set to: $token_lim"

        vllm serve "$model"  --chat-template ../examples/template_chatml.jinja \
                --port 8000 --batched-mode \
                --max_num_seqs "$1" \
                --prefill_batch_size "$1" \
                --max_num_batched_tokens "$token_lim" \
                --model-stats-log-dir "$log_dir" &
}
for ((i=1;i<100;i*=2)); do
        start_server "$i"
        sleep 40 #Sleep to ensure server startup is complete
        sudo nvidia-smi --lock-gpu-clocks=1410,1410 #Change GPU mem and SM frequencies here
        sudo nvidia-smi --lock-memory-clocks=5001,5001
        #Run benchmark
        python benchmark_serving.py --backend vllm --model "$model" --dataset-name random --num_prompts "$i" --random-input-len "$input_len" --random-output-len 100 --experiment-mode BATCHED --batch_size "$i"

        mv "$log_dir"/temp.txt temp.txt
        if [[ ! -f "temp_old.txt" ]];then
                mv temp.txt temp_old.txt
        else
                paste -d',' temp_old.txt temp.txt > res.txt
        fi
        mv res.txt temp_old.txt
        #Kill server process
        kill -SIGTERM "$!"
done
rm temp.txt
mv temp_old.txt res.txt


