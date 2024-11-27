import argparse
import random
import time
from typing import List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    p_geometric: Optional[float] = None
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    if p_geometric is not None:
        output_lens = [min(np.random.geometric(p_geometric), 2048 - x)
                       for x in input_lens]
    else:
        output_lens = np.random.randint(
            int(output_len * range_ratio),
            output_len + 1,
            size=num_prompts,
        )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests


def benchmark(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    prefill_k: int,
) -> None:
    from vllm import SamplingParams, LLM
    prompts = [x[0] for x in input_requests]
    sampling_params = [SamplingParams(n=1,
                                      temperature=1.0,
                                      top_p=1.0,
                                      ignore_eos=True,
                                      max_tokens=x[2]) for x in input_requests]
    llm = LLM(model=model_id, max_num_seqs=100,
              prefill_batch_size=prefill_k, max_num_batched_tokens=20000)
    benchmark_start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    benchmark_duration = time.perf_counter() - benchmark_start_time
    with open('log_tpt_new.txt', 'a') as f:
        f.write(f'{len(input_requests) / benchmark_duration}\n')


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    input_requests = sample_random_requests(
        prefix_len=args.random_prefix_len,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        num_prompts=args.num_prompts,
        range_ratio=args.random_range_ratio,
        tokenizer=tokenizer,
        p_geometric=args.p_geometric
    )

    benchmark(
        model_id=model_id,
        tokenizer=tokenizer,
        input_requests=input_requests,
        prefill_k=args.prefill_k
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--prefill-k",
        type=int,
        default=1,
        help='Set scheduler prefill to k.'
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
    )
    # group for dataset specific arguments
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")
    random_group.add_argument(
        "--p-geometric",
        type=float,
        default=None,
        help="Use geometric output token lens")

    args = parser.parse_args()
    main(args)
