
import threading
import time

class _BaseWatcherThread(threading.Thread):
    def __init__(self, interval_secs=1.0, **kwargs):
        super().__init__(daemon=True)
        assert interval_secs >= 1.0
        self._interval_secs = interval_secs
        self._iteration = 0
        self._update()
        self.start()

    @property
    def iteration(self):
        return self._iteration

    def _update(self):
        raise NotImplementedError

    def run(self):
        while True:
            time.sleep(self._interval_secs)
            self._update()
            self._iteration += 1

