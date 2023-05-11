import torch
import logging

LOGGER = logging.getLogger(__name__)

class ExponentialMovingAverage(torch.nn.Module):
    def __init__(self, init_module, mu, data_parallel=False):
        super(ExponentialMovingAverage, self).__init__()
        self.module = init_module
        # self.module = copy.deepcopy(init_module)
        # self.module.requires_grad_(False)

        self.mu = mu
        self.data_parallel = data_parallel

    def forward(self, x):
        m = torch.nn.parallel.DataParallel(self.module) if self.data_parallel else self.module
        return m(x)

    def update(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            # see : https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/ExponentialMovingAverage?hl=PL
            mu = min(self.mu, (1. + step) / (10. + step))

        state_dict = {}
        with torch.no_grad():
            for (name, m1), (name2, m2) in zip(self.module.state_dict().items(), module.state_dict().items()):
                if name != name2:
                    LOGGER.warning('[ExpoentialMovingAverage] not matched keys %s, %s', name, name2)

                if step is not None and step < 0:
                    state_dict[name] = m2.clone().detach()
                else:
                    state_dict[name] = ((mu * m1) + ((1.0 - mu) * m2)).clone().detach()

        self.module.load_state_dict(state_dict)