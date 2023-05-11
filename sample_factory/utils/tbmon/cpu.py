
import psutil

from ._base import _BaseWatcherThread

class CpuWatcher(_BaseWatcherThread):
    def __init__(self, interval_secs=1.0):
        self._utilization = 0
        self._vm_info = None
        self._swap_info = None
        super().__init__(interval_secs)

    @property
    def count(self):
        return psutil.cpu_count()

    @property
    def utilization(self):
        return self._utilization

    @property
    def memory_percent(self):
        return self._vm_info.percent

    @property
    def memory_used_bytes(self):
        return self._vm_info.used

    @property
    def memory_used_mbytes(self):
        return self.memory_used_bytes / 1024 / 1024

    @property
    def memory_used_gbytes(self):
        return self.memory_used_mbytes / 1024

    def _update(self):
        self._utilization = psutil.cpu_percent()
        self._vm_info = psutil.virtual_memory()
        self._swap_info = psutil.swap_memory()
