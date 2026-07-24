from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass

import psutil

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class ResourceProfile:
    wall_time_sec: float
    rss_start_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int
    rss_peak_delta_bytes: int
    vms_peak_bytes: int
    private_peak_bytes: int
    gpu_allocated_peak_bytes: int | None
    gpu_reserved_peak_bytes: int | None
    sample_interval_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    """Sample process-tree RAM and PyTorch CUDA peaks for one experiment."""

    def __init__(self, sample_interval_sec: float = 0.1):
        self.sample_interval_sec = float(sample_interval_sec)
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_start = 0
        self._rss_end = 0
        self._rss_peak = 0
        self._vms_peak = 0
        self._private_peak = 0
        self._start = 0.0

    def _tree_rss(self) -> int:
        return self._tree_memory()[0]

    def _tree_memory(self) -> tuple[int, int, int]:
        """Return process-tree RSS, virtual size, and private/USS memory."""
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except psutil.Error:
            pass
        rss_total = 0
        vms_total = 0
        private_total = 0
        for process in processes:
            try:
                info = process.memory_info()
                rss_total += int(info.rss)
                vms_total += int(info.vms)
                try:
                    full = process.memory_full_info()
                    private_total += int(getattr(full, "uss", getattr(full, "private", info.rss)))
                except psutil.Error:
                    private_total += int(info.rss)
            except psutil.Error:
                continue
        return rss_total, vms_total, private_total

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_sec):
            rss, vms, private = self._tree_memory()
            self._rss_peak = max(self._rss_peak, rss)
            self._vms_peak = max(self._vms_peak, vms)
            self._private_peak = max(self._private_peak, private)

    def __enter__(self) -> "ResourceMonitor":
        self._rss_start, self._vms_peak, self._private_peak = self._tree_memory()
        self._rss_peak = self._rss_start
        self._start = time.perf_counter()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.sample_interval_sec))
        self._rss_end, vms_end, private_end = self._tree_memory()
        self._rss_peak = max(self._rss_peak, self._rss_end)
        self._vms_peak = max(self._vms_peak, vms_end)
        self._private_peak = max(self._private_peak, private_end)

    def result(self) -> ResourceProfile:
        gpu_allocated = None
        gpu_reserved = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_allocated = int(torch.cuda.max_memory_allocated())
            gpu_reserved = int(torch.cuda.max_memory_reserved())
        return ResourceProfile(
            wall_time_sec=time.perf_counter() - self._start,
            rss_start_bytes=self._rss_start,
            rss_end_bytes=self._rss_end,
            rss_peak_bytes=self._rss_peak,
            rss_peak_delta_bytes=max(0, self._rss_peak - self._rss_start),
            vms_peak_bytes=self._vms_peak,
            private_peak_bytes=self._private_peak,
            gpu_allocated_peak_bytes=gpu_allocated,
            gpu_reserved_peak_bytes=gpu_reserved,
            sample_interval_sec=self.sample_interval_sec,
        )
