
import time
from datetime import datetime

from .cpu import CpuWatcher
from .gpu import GpuWatcher
from .network import NetworkWatcher

class Watcher(object):
    def __init__(self, interval_secs=3.0, enable_gpu=True, enable_cpu=True, enable_network=True):
        assert interval_secs > 0.5
        self._interval_secs = interval_secs

        self._start_time = time.time()
        self._network_watcher = NetworkWatcher() if enable_network else None
        self._cpu_watcher = CpuWatcher() if enable_cpu else None
        self._gpu_watcher = GpuWatcher() if enable_gpu else None

    @property
    def cpu(self) -> CpuWatcher:
        """

        :return:
        """
        return self._cpu_watcher

    @property
    def network(self) -> NetworkWatcher:
        """

        :return:
        """
        return self._network_watcher

    @property
    def gpu(self) -> GpuWatcher:
        """

        :return:
        """
        return self._gpu_watcher

    @property
    def summary(self):
        return {
            'TIME': datetime.now(),
            'GPU': {
                'count': len(self.gpu.gpu_names),
                'info': self.gpu.gpus_info,
            },
            'CPU': {
                'count': self.cpu.count,
                'utilization': self.cpu.utilization,
                'memory': self.cpu.memory_percent,
            },
            'NETWORK': self.network.traffic_per_nic()
        }

    def print_summary(self) -> None:
        """

        :return:
        """
        print('{} {} {}'.format('-' * 10, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 50))
        print('CPU\t{:}-cores\tUTIL:{:.1f}\tMEM:{:.1f}%'.format(
            self.cpu.count, self.cpu.utilization, self.cpu.memory_percent))
        for gpu in self.gpu.gpus_info:
            print('GPU\t{:12s}\tUTIL:{:.1f}%\tMEM:{:.1f}%'.format(gpu.name, gpu.gpu_rate, gpu.mem_rate))
        print('NIC\t{:12s}\tRECV:{}MB/s\tSENT:{}MB/s'.format(
            self.network.nic_eth0.nic_name, self.network.nic_eth0.recv_mbytes, self.network.nic_eth0.sent_mbytes))
