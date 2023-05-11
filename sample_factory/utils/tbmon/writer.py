
import threading
import logging
import time

from tensorboardX import SummaryWriter

from .watcher import Watcher

class SummaryWriterThread(threading.Thread):
    def __init__(self, writer, interval_secs=5.0, enable_gpu=True, enable_cpu=True, enable_network=True):
        super().__init__(daemon=True)
        assert isinstance(writer, SummaryWriter)
        self._summary_writer = writer
        self._interval_secs = interval_secs
        self._iteration = 0
        self._enable_gpu = enable_gpu
        self._enable_cpu = enable_cpu
        self._enable_network = enable_network
        self._watcher = Watcher()
        self._set_custom_scalars_layout()

    def _set_custom_scalars_layout(self):
        """

        :return:
        """
        custom_scalars_layout = {
            'gpu': {
                'util': [
                    'Multiline', [
                        'GPU/util/{}'.format(x) for x in self._watcher.gpu.gpu_names
                    ]
                ],
                'memory': [
                    'Multiline', [
                        'GPU/memory/{}'.format(x) for x in self._watcher.gpu.gpu_names
                    ]
                ]
            },
            'network': {
                'eth0': [
                    'Multiline', [
                        'NETWORK/{}/sent'.format(self._watcher.network.nic_eth0.nic_name),
                        'NETWORK/{}/recv'.format(self._watcher.network.nic_eth0.nic_name)
                    ]
                ]
            }
        }
        if hasattr(self._summary_writer, 'add_custom_scalars'):  # tensorboardX >= 1.5
            self._summary_writer.add_custom_scalars(custom_scalars_layout)

    def run(self):
        """

        :return:
        """
        while True:
            if self._enable_gpu:
                for gpu in self._watcher.gpu.gpus_info:
                    self._summary_writer.add_scalar(
                        'monitors/gpu/util/{}'.format(gpu.name), gpu.gpu_rate, self._iteration)
                    self._summary_writer.add_scalar(
                        'monitors/gpu/memory/{}'.format(gpu.name), gpu.mem_rate, self._iteration)
                    logging.debug('{:} gpu:{:.1f}% mem:{:.1f}%({:.1f}GB)'.format(
                        gpu.name, gpu.gpu_rate, gpu.mem_rate, gpu.mem_used_gbytes))

            if self._enable_cpu:
                self._summary_writer.add_scalar('monitors/cpu/util', self._watcher.cpu.utilization, self._iteration)
                self._summary_writer.add_scalar('monitors/cpu/memory', self._watcher.cpu.memory_percent, self._iteration)
                logging.debug('cpu:{:.3f} mem:{:.3f}%({:.1f}GB)'.format(
                    self._watcher.cpu.utilization,
                    self._watcher.cpu.memory_percent,
                    self._watcher.cpu.memory_used_gbytes))

            if self._enable_network:
                eth0 = self._watcher.network.nic_eth0
                self._summary_writer.add_scalar('monitors/network/{}/sent'.format(eth0.nic_name),
                                                eth0.sent_mbytes, self._iteration)
                self._summary_writer.add_scalar('monitors/network/{}/recv'.format(eth0.nic_name),
                                                eth0.recv_mbytes, self._iteration)
                logging.debug('network: sent:{:.3f}mb/s recv:{:.3f}mb/s'.format(
                    eth0.sent_mbytes, eth0.recv_mbytes))

            time.sleep(self._interval_secs)
            self._iteration += 1
