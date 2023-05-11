import os
import collections

import torch
from sample_factory.utils.utils import log
from tensorboardX import SummaryWriter
from sample_factory.utils.utils import AttrDict

DistEnv = collections.namedtuple('DistEnv', ['world_size', 'world_rank', 'local_rank', 'num_gpus', 'master'])

def dist_init(cfg):
    if int(os.environ['WORLD_SIZE']) > 1:
        log.debug('[dist] Distributed: wait dist process group:%d', cfg.local_rank)
        torch.distributed.init_process_group(backend=cfg.dist_backend, init_method='env://',
                                world_size=int(os.environ['WORLD_SIZE']))
        assert (int(os.environ['WORLD_SIZE']) == torch.distributed.get_world_size())
        log.debug('[dist] Distributed: success device:%d (%d/%d)',
                    cfg.local_rank, torch.distributed.get_rank(), torch.distributed.get_world_size())
        distenv = DistEnv(torch.distributed.get_world_size(), torch.distributed.get_rank(), cfg.local_rank, 1, torch.distributed.get_rank() == 0)
    else:
        log.debug('[dist] Single processed')
        distenv = DistEnv(1, 0, 0, torch.cuda.device_count(), True)
    log.debug('[dist] %s', distenv)
    return distenv

def dist_all_reduce_gradient(model):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    for p in model.parameters():
        if type(p.grad) is not type(None):
            torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
            p.grad.data /= world_size

def dist_reduce_gradient(model, grads=None):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    if grads is None:
        for p in model.parameters():
            if type(p.grad) is not type(None):
                torch.distributed.reduce(p.grad.data, 0, op=torch.distributed.ReduceOp.SUM)
                p.grad.data /= world_size
    else:
        for grad in grads:
            if type(grad) is not type(None):
                torch.distributed.reduce(grad.data, 0, op=torch.distributed.ReduceOp.SUM)
                grad.data /= world_size


def dist_all_reduce_buffers(model):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    #world_size = torch.distributed.get_world_size()

    # for n, b in model.named_buffers():
    #     if b.type() != 'torch.cuda.FloatTensor':
    #         print(n)
    #         print(b.type())

    for n, b in model.named_buffers():
        if b.type() != 'torch.cuda.LongTensor':
            torch.distributed.all_reduce(b.data, op=torch.distributed.ReduceOp.SUM)
            b.data /= world_size

def dist_broadcast_model(model):
    torch.distributed.barrier()
    for _, param in model.state_dict().items():
        torch.distributed.broadcast(param, 0)
    torch.distributed.barrier()
    torch.cuda.synchronize()


TENSORBOARD_GLOBAL_STATS = dict()

class AsyncDistSummaryWrapper():
    def __init__(self, summary_dir, key=None, default_scalar_op=None, default_step_op=None, **kwargs):
        self.default_scalar_op = default_scalar_op
        self.default_step_op = default_step_op
        self.is_dist = False
        self.is_master = False
        self.key = key

        current_env = os.environ.copy()
        if int(current_env.get('WORLD_SIZE', '1')) > 1:
            self.is_dist = True
        if int(current_env.get('RANK', '0')) == 0:
            self.is_master = True
            self.writer = SummaryWriter(summary_dir, **kwargs)

        if self.is_dist:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])

            self.rpc_name = 'master' if self.is_master else f'worker_{self.rank}'

            os.environ['MASTER_PORT'] = os.environ["MASTER_EXTRA_PORT"]
            backend = torch.distributed.rpc.BackendType.PROCESS_GROUP
            rpc_backend_options = torch.distributed.rpc.ProcessGroupRpcBackendOptions(rpc_timeout=60000)
            torch.distributed.rpc.init_rpc(self.rpc_name, backend=backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)

            os.environ['MASTER_PORT'] = current_env['MASTER_PORT']

            if self.is_master:
                global TENSORBOARD_GLOBAL_STATS
                TENSORBOARD_GLOBAL_STATS[self.key] = [dict() for _ in range(self.world_size)]

    @classmethod
    def set_global_stats(cls, key, rank, tag, stats):
        TENSORBOARD_GLOBAL_STATS[key][rank][tag] = stats

    def get_global_stats(self, tag):
        return [TENSORBOARD_GLOBAL_STATS[self.key][w].get(tag, None) for w in range(self.world_size)]

    def terminate(self):
        if self.is_dist:
            torch.distributed.rpc.shutdown()

    def add_scalar(self, tag, scalar_value, global_step=None, scalar_op=None, step_op=None):
        """
        Wrapper function that collects stats (tag, scalar_value, tag) from workers and plots.
        """
        scalar_op = self.default_scalar_op if scalar_op is None else scalar_op  # overwrite
        step_op = self.default_step_op if step_op is None else step_op  # overwrite

        if self.is_dist and (scalar_op in ['avg', 'sum', 'max', 'min'] or step_op in ['avg', 'sum']):
            # scalar_value = scalar_value if torch.is_tensor(scalar_value) else torch.tensor(scalar_value)
            # global_step = global_step if torch.is_tensor(global_step) else torch.tensor(global_step)
            stats = dict(scalar_value=scalar_value, global_step=global_step)
            if not self.is_master:
                torch.distributed.rpc.rpc_sync('master', AsyncDistSummaryWrapper.set_global_stats, args=(self.key, self.rank, tag, stats))
            else:
                self.set_global_stats(self.key, self.rank, tag, stats)
                global_stats = self.get_global_stats(tag)

                if scalar_op is None:
                    pass
                elif scalar_op in ['max', 'min']:
                    for s in global_stats:
                        if s is not None:
                            if scalar_op == 'max':
                                scalar_value = max(scalar_value, s['scalar_value'])
                            elif scalar_op == 'min':
                                scalar_value = min(scalar_value, s['scalar_value'])
                            else:
                                scalar_value += s['scalar_value']
                elif scalar_op in ['avg', 'sum']:
                    scalar_value = 0
                    scalar_count = 0
                    for s in global_stats:
                        if s is not None:
                            scalar_value += s['scalar_value']
                            scalar_count += 1
                    if scalar_op == 'avg':
                        scalar_value = scalar_value / scalar_count
                    elif scalar_op == 'sum':
                        scalar_value = scalar_value / scalar_count * self.world_size
                else:
                    raise NotImplementedError

                if step_op is None:
                    pass
                elif step_op in ['avg', 'sum']:
                    global_step = 0
                    step_count = 0
                    for s in global_stats:
                        if s is not None:
                            global_step += s['global_step']
                            step_count += 1
                    if step_op == 'avg':
                        global_step = global_step / step_count
                    elif step_op == 'sum':
                        global_step = global_step / step_count * self.world_size
                else:
                    raise NotImplementedError

                self.writer.add_scalar(tag, scalar_value, global_step)
        else:
            if self.is_master:
                self.writer.add_scalar(tag, scalar_value, global_step)

