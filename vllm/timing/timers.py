import torch.cuda
from typing import Optional
import time
"""
A context manager for timing torch kernel executions and CPU exec times for a block of code.
Units of time in ms
"""


class CudaTimer:
    def __init__(self, op: str, enabled: bool, sync_after_exec: bool = True):
        self.enabled = enabled
        self.op = op
        self.sync_after_exec = sync_after_exec
        self.timing_value: Optional[float] = None

    def __enter__(self):
        if not self.enabled:
            return
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        if self.sync_after_exec:
            torch.cuda.synchronize()
            self.set_exec_time()

    def set_exec_time(self):
        # WARN: Should only be called after call to torch.cuda.synchronize
        if not self.enabled:
            return
        self.timing_value = self.start_event.elapsed_time(self.end_event)


class CPUTimer:
    def __init__(self, op: str, enabled: bool):
        self.enabled = enabled
        self.op = op
        self.timing_value: Optional[float] = None

    def __enter__(self):
        if not self.enabled:
            return
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        self.timing_value = (time.perf_counter() - self.start_time) * 1e-3
