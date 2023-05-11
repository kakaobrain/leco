
import platform

import psutil
import time

from ._base import _BaseWatcherThread

class _NetworkStatus(object):
    def __init__(self, name: str, time_secs: float, sent_bytes: int, recv_bytes: int):
        self._name = name
        self._time_secs = time_secs
        self._sent_bytes = sent_bytes
        self._recv_bytes = recv_bytes

    @property
    def time_secs(self) -> float:
        return self._time_secs

    @property
    def bytes_sent(self) -> float:
        return self._sent_bytes

    @property
    def bytes_recv(self) -> float:
        return self._recv_bytes

    @property
    def name(self) -> str:
        return self._name


class _TrafficStatus(object):
    def __init__(self, nic_name: str, prev: _NetworkStatus, curr: _NetworkStatus):
        assert isinstance(prev, _NetworkStatus) or prev is None
        assert isinstance(curr, _NetworkStatus)
        self._nic_name = nic_name
        self._prev = prev
        self._curr = curr

    @property
    def nic_name(self) -> str:
        return self._nic_name

    @property
    def _elapsed(self) -> float:
        if self._prev is None:
            return 0.0
        return self._curr.time_secs - self._prev.time_secs

    @property
    def sent_bytes(self) -> float:
        if self._prev is None:
            return 0.0
        return (self._curr.bytes_sent - self._prev.bytes_sent) // self._elapsed

    @property
    def recv_bytes(self) -> float:
        if self._prev is None:
            return 0.0
        return (self._curr.bytes_recv - self._prev.bytes_recv) // self._elapsed

    @property
    def recv_mbytes(self) -> float:
        return self.recv_bytes / 1024 / 1024

    @property
    def sent_mbytes(self) -> float:
        return self.sent_bytes / 1024 / 1024

    @property
    def prev_status(self) -> _NetworkStatus:
        return self._prev

    @property
    def curr_status(self) -> _NetworkStatus:
        return self._curr

    def __repr__(self):
        return '{}(nic:{}, recv:{:.3f}MB/s, sent:{:.3f}MB/s)'.format(
            self.__class__.__name__, self.nic_name, self.recv_mbytes, self.sent_mbytes)


class NetworkWatcher(_BaseWatcherThread):
    def __init__(self, interval_secs: float = 1.0):
        self._network_traffic_dict = dict()
        super().__init__(interval_secs)

    def _update(self):
        net_io_counters = psutil.net_io_counters(pernic=True)
        for name, counters in net_io_counters.items():
            nic_traffic = self._network_traffic_dict.get(name)
            prev_status = nic_traffic.curr_status if nic_traffic else None
            curr_status = _NetworkStatus(name, time.time(), counters.bytes_sent, counters.bytes_recv)
            self._network_traffic_dict[name] = _TrafficStatus(name, prev_status, curr_status)

    @property
    def interface_names(self):
        return self._network_traffic_dict.keys()

    @property
    def nic_eth0(self):
        nic_name = None
        os_name = platform.system()
        if os_name == 'Linux':  # for linux
            nic_name = 'eth0'
        elif os_name == 'Darwin':  # for mac
            nic_name = 'en0'
        return self._network_traffic_dict.get(nic_name)

    def traffic(self, interface_name: str) -> _TrafficStatus:
        return self._network_traffic_dict.get(interface_name)

    def traffic_per_nic(self) -> dict:
        return self._network_traffic_dict
