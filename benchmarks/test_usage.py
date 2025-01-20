from vllm import LLM
from vllm.utils import FlexibleArgumentParser


# Create an LLM with desired params to see memory profile
def main(args):
    llm = LLM(model=args.model,
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              gpu_memory_utilization=args.mem_util  # Use all of GPU
              )


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Profile model memory usage',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of model.'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=2048,
        help='Max model len.'
    )
    parser.add_argument(
        '--max-num-seqs',
        type=int,
        default=100,
        help='Max number of seqs for parallel decode.'
    )
    parser.add_argument(
        '--max-num-batched-tokens',
        type=int,
        default=2048,
        help='Max no of tokens for parallel prefill/decode.'
    )
    parser.add_argument(
        '--mem-util',
        type=float,
        default=0.95,
        help='GPU mem util between (0 and 1]'
    )
    main(parser.parse_args())
