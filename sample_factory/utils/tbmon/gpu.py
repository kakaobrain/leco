
from typing import List

import py3nvml.py3nvml as pynvml

from ._base import _BaseWatcherThread

class _GpuStatus(object):
    def __init__(self, name, gpu_rate, mem_used_bytes, mem_total_bytes):
        self._name = name
        self._gpu_rate = gpu_rate
        self._mem_used_bytes = mem_used_bytes
        self._mem_total_bytes = mem_total_bytes

    @property
    def name(self) -> str:
        return self._name

    @property
    def gpu_rate(self) -> float:
        return self._gpu_rate

    @property
    def mem_used_bytes(self) -> float:
        return self._mem_used_bytes

    @property
    def mem_used_mbytes(self):
        return self.mem_used_bytes / 1024 / 1024

    @property
    def mem_used_gbytes(self):
        return self.mem_used_mbytes / 1024

    @property
    def mem_rate(self):
        return 100.0 * self._mem_used_bytes / self._mem_total_bytes

    def __repr__(self):
        return '[GPU-{:}] util:{:.1f}% mem:{:.1f}%({:.1f}GB)'.format(
            self.name, self.gpu_rate, self.mem_rate, self._mem_used_bytes / 1024 / 1024 / 1024)


class GpuWatcher(_BaseWatcherThread):
    def __init__(self, interval_secs=1.0):
        pynvml.nvmlInit()
        self._device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        super().__init__(interval_secs)

    @property
    def gpu_count(self) -> int:
        return pynvml.nvmlDeviceGetCount()

    @property
    def gpu_names(self) -> List[str]:
        names = list()
        for i in range(self.gpu_count):
            handle = self._device_handles[i]
            gpu_name = pynvml.nvmlDeviceGetName(handle).replace(' ', '_')
            names.append('{}/{}'.format(gpu_name, i))
        return names

    @property
    def gpus_info(self) -> List[_GpuStatus]:
        gpus = list()
        for i in range(self.gpu_count):
            handle = self._device_handles[i]
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append(_GpuStatus(self.gpu_names[i], gpu_util_rates.gpu, memory_info.used, memory_info.total))
        return gpus

    def _update(self):
        return
