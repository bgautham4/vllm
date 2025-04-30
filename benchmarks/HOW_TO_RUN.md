# Pytorch profiler benchmarking

## Installing vLLM
### Setup environment
It is recommended to create a `python-venv` or `conda` environment for ease of use.
#### venv
```bash
# Ensure you are using python-3.9 or above and you have the *venv* module
python -m venv ./.venv
# Activate environment
source .venv/bin/activate
```
#### conda
Follow directions to install `miniconda` from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)
```bash
# Create conda env
conda create -n vllm_profiling python=3.12 -y
conda activate vllm_profiling
```
### Install vLLM
Check CUDA runtime version by running `nvidia-smi`. If your CUDA version is below 12.6, then you will have to either install [pytorch](https://pytorch.org/get-started/locally/) for your CUDA version and install vLLM using [your exsiting torch version](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda) or update your NVIDIA driver to a version supporting the runtime for CUDA>=12.6. 
```bash
# If CUDA>=12.6
git clone https://github.com/bgautham4/vllm.git -b latest-profiler
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
---
## Profiling scripts
Profiling scripts are contained in `vllm/benchmarks/`

1. decode_exp.sh: Script for profiling controlled batching of decodes.
2. prefill_exp.sh: Script for profiling controlled batching of prefills.
### Run experiments
```bash
ulimit -n 65536 # Increase open fd limit for current session
./run_exp.sh
```
### Parse results
Parsing scripts are found in `vllm/benchmarks/parsing_scripts`
```
./parse_traces.sh
# Directory should now contain kernel_dat.csv, op_times.tsv and op_times_detailed.tsv
```
---
## Methodology
### Controlled batching with custom scheduler policy
```python
# In vllm/vllm/core/scheduler.py line:1325
def _my_schedule(self) -> SchedulerOutputs:
        """Schedule queued requests.
        ...
         if not self.swapped:
            if (budget.max_num_seqs - budget.num_curr_seqs >= self.scheduler_config.prefill_batch_size):
                # Schedule prefills if there is "occupancy" >= "prefill_batch_size"
                prefills = self._schedule_prefills(budget,
                                                   curr_loras,
                                                   enable_chunking=False)

		...
```
### PyTorch profiler for profiling model forward
```python
# In vllm/vllm/worker/model_runner.py line:1774
if (profile_now):
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                 record_shapes=True) as p:
                        hidden_or_intermediate_states = model_executable(
                            input_ids=model_input.input_tokens,
                            positions=model_input.input_positions,
                            intermediate_tensors=intermediate_tensors,
                            **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                         device=self.device),
                            **seqlen_agnostic_kwargs,
                            **model_kwargs,
                        )
                    print(p.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=10))
                    p.export_chrome_trace(
                        "./trace_" + str(self.step_num) + ".json")
                    self.step_num += 1
```