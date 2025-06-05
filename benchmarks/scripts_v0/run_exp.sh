#!/bin/bash

huggingface-cli login --token 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
# Download and cache llama model weights
python ../examples/offline_inference/llm_engine_example.py --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 --max-num-seqs 10 || exit 1
#Start experiment
./decode_exp.sh --model meta-llama/Llama-3.1-8B-Instruct
