from vllm.timing.timers import CudaTimer
from functools import wraps
from vllm.logger import init_logger
from time import perf_counter
import inspect

logger = init_logger(__name__)

"""
To simplify timing of CUDA events on a layer by layer basis 
Example:

class NN(nn.Module):
    ...
    @log_cuda_timer(op="NN_FORWARD")
    def forward(...):
"""
def log_cuda_timer(op: str):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    logger.info("\x1b[1;4;93mCUDA timer placed in \x1b[92m%s\x1b[93m at line \x1b[92m%d\x1b[93m for op %s\x1b[0m", filename, lineno, op)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with CudaTimer(op=op, enabled=True, sync_after_exec=True) as timer:
                out = func(*args, **kwargs)
            logger.trace("CUDA_TIMER", extra={"op": op, "ts": perf_counter(), "time_taken_ms": timer.timing_value})
            return out
        return wrapper

    return decorator

"""
To simplify profiling of CUDA and CPU events on a layer by layer basis 

Makes use of the pytorch profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
Example:

class NN(nn.Module):
    ...
    @profile_cuda_event(op="NN_FORWARD")
    def forward(...):
"""
def profile_cuda_event(op: str):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    logger.info("\x1b[1;4;93mProfiler placed in \x1b[92m%s\x1b[93m at line \x1b[92m%d\x1b[93m for op %s\x1b[0m", filename, lineno, op)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if (not hasattr(wrapper, 'cntr')):
                wrapper.cntr = 1
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
                out = func(*args, **kwargs)
            logger.trace("PROF", extra={"saved_to": f'trace_{wrapper.cntr}.json'})
            p.export_chrome_trace(
                    "./traces/trace_" + str(wrapper.cntr) + ".json")
            wrapper.cntr += 1
            return out
        return wrapper

    return decorator
