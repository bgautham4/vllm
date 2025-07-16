"""
A context manager for timing torch kernel executions and CPU exec times for a block of code.
Units of time in ms
"""
import torch.cuda
from typing import Optional
from time import perf_counter

class CudaTimer:
    def __init__(self, op: str, enabled: bool):
        self.enabled = enabled
        self.op = op
        self.timing_value: Optional[float] = None

    def __enter__(self):
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        torch.cuda.synchronize()
        self.timing_value = self.start_event.elapsed_time(self.end_event)

class CPUTimer:
    def __init__(self, op: str, enabled: bool):
        self.enabled = enabled
        self.op = op
        self.timing_value: Optional[float] = None

    def __enter__(self):
        if self.enabled:
            self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        self.timing_value = (perf_counter() - self.start_time) * 1e3 #In ms
