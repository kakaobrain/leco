import numpy as np
import copy
import torch
from torch import nn
from torchvision import transforms
from dataclasses import dataclass
from typing import List
import random
import math


DMLAB_OBS_SHAPE = (72, 96)
MINIGRID_OBS_SHAPE = (7, 7)

@dataclass(repr=True)
class CellCount:
    cnt: int
    best_epi_cnt: int
    steps: int

    def __init__(self, cnt, best_epi_cnt=0, steps=0):
        self.cnt = cnt
        self.best_epi_cnt = best_epi_cnt
        self.steps = steps


@dataclass(init=True)
class CellCountList:
    items: List[CellCount]
    cells: np.ndarray

    def counts_in_this_episode(self):
        result = [item.cnt for item in self.items]
        return result

    def steps_in_this_episode(self):
        result = [item.steps for item in self.items]
        return result

    def cell_keys(self):
        result = []
        for i in range(len(self.cells)):
            cell = self.cells[i]
            result.append(CellCounter.cell2key(cell))
        return result


@dataclass(repr=True)
class CellSpec:
    type: str
    offset: tuple
    resolution: tuple
    depth: int
    vq_type: str
    num_res_blocks: int
    num_cnn_blocks: int

    def __eq__(self, other):
        return self.resolution == other.resolution and self.depth == other.depth

    def parse_resolution(self, res: str, is_ds_cell=False):
        h, w, d = tuple(map(int, res.split('x')))
        self.resolution = (h, w)
        self.depth = d
        if is_ds_cell:
            self.depth -= 1

    @property
    def resize_target_shape(self):
        proportion = 2 ** self.num_cnn_blocks
        target_shape = (self.resolution[0] * proportion, self.resolution[1] * proportion)
        return target_shape

    def __init__(self, spec: str, type=None, is_ds_cell=False):
        self.type = type
        self.vq_type = None

        if type is not None:
            s = type.split('_')[0]
            try:
                self.num_res_blocks = int(s[-1])
            except:
                self.num_res_blocks = 0
            self.vq_type = ''.join([i for i in s if not i.isdigit()])

        items = [item.strip() for item in spec.split(',')]
        assert len(items) == 1 or len(items) == 3
        if len(items) == 3:
            self.offset = (int(items[0]), int(items[1]))
        else:
            self.offset = None
        self.parse_resolution(items[-1], is_ds_cell=is_ds_cell)

        self.num_cnn_blocks = 4
        if self.type.startswith('vqr') or self.type.startswith('aer'):
            if self.resolution == (4,6):
                self.num_cnn_blocks = 4
            elif self.resolution == (2,3):
                self.num_cnn_blocks = 5


class Cells:
    def __init__(self):
        self.hash = {}
        self._deleted = {}
        self.hash_keys = ['actor_id', 'split_id', 'env_id', 'episode_id']

    def get_key(self, stat):
        key = ()
        for _key in self.hash_keys:
            if type(stat[_key]) == tuple:
                key += stat[_key]
            else:
                key += (stat[_key],)
        return key

    def _prepare_to_add_cell(self, stat):
        pass

    def finalize(self, stat):
        pass

    def has_removed(self, stat):
        key = self.get_key(stat)
        return key in self._deleted

    def prob_cell(self, hash, cell_id):
        cnts = []
        idx = None

        score = None
        if 'score' in hash:
            score = hash['score']
            del hash['score']

        for i, (_id, cnt) in enumerate(hash.items()):
            if idx is None and cell_id == _id:
                idx = i
            cnts.append(cnt)

        if score is not None:
            hash['score'] = score

        if idx is None:
            prob = 0
        else:
            n_total = np.sum(cnts)
            if n_total == 0:
                prob = 0
            else:
                probs = np.array(cnts)/n_total
                prob = probs[idx]

        return prob

    def add(self, stat, cell_id):
        _hash = self._prepare_to_add_cell(stat)
        if not cell_id in _hash:
            _hash[cell_id] = 0
        out = CellCount(_hash[cell_id])
        _hash[cell_id] += 1

        if stat['done']:
            # print("1:", stat)
            self.finalize(stat)

        return out


@dataclass
class EpisodeItem:
    score: float
    steps: int
    cells: dict

    def __init__(self, score=0.0, steps=0, cells=None):
        self.score = score
        self.steps = steps
        self.cells = cells


class EpisodeItemList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = np.array([])
        self.ma = 0
        self.n = 0

    def running_mean(self, score):
        ma = self.ma
        if self.n == 0:
            prev_sum = 0
        else:
            prev_sum = self.ma * self.n
        self.n += 1
        self.ma = (prev_sum + score) / self.n
        return ma

    def append(self, item: EpisodeItem):
        self.items = np.append(self.items, item)
        # TODO: heapfy?
        if len(self.items) > self.max_size:
            scores = [item.score for item in self.items]
            s_ids = np.argsort(scores)
            self.items = self.items[s_ids][-self.max_size:]

    def avg_score(self):
        if len(self.items) == 0:
            return 0
        scores = [item.score for item in self.items]
        return np.mean(scores)

    def __len__(self):
        return len(self.items)

    def count(self, cell_id):
        cnts = []
        for item in self.items:
            cnt = 0
            if cell_id in item.cells:
                cnt = item.cells[cell_id]
            cnts.append(cnt)
        return math.floor(np.mean(cnts))


class EpisodicCells(Cells):
    def __init__(self, with_best_episode=0):
        self.hash = {}
        self._deleted = {}
        self.hash_keys = ['task_id', 'index', 'episode_id']
        self.task_hash = {}
        self.with_best_episode = with_best_episode
        if with_best_episode:
            self.minimum_score = 0

    def _prepare_to_add_task(self, task_id):
        if self.with_best_episode and not task_id in self.task_hash:
            self.task_hash[task_id] = EpisodeItemList(self.with_best_episode)

    def _prepare_to_add_cell(self, stat):
        key = self.get_key(stat)
        if not key in self.hash:
            self.hash[key] = {}

        _hash = self.hash[key]

        # after an episode is ended (i.e. stat['done'] == True), hash corresponded to the episode will be removed.
        # if a request which queries the removed hash is exist, it means that there will be a bug.
        assert not self.has_removed(stat)

        if not 'score' in _hash:
            _hash['score'] = 0
            _hash['steps'] = 0

        return _hash

    def finalize(self, stat):
        key = self.get_key(stat)

        if self.with_best_episode:
            task_id = stat['task_id']
            current_score = self.hash[key]['score']
            current_steps = self.hash[key]['steps']
            # episodic_score = self.task_hash[task_id].avg_score()
            episodic_score = self.task_hash[task_id].running_mean(current_score)

            should_update = current_score > self.minimum_score and episodic_score <= current_score
            if should_update:
                print(f'updated episodic score of task-{task_id} from {episodic_score} to {current_score} at {current_steps} steps')
                print('\t', stat)
                new_item = EpisodeItem(current_score, current_steps, copy.deepcopy(self.hash[key]))
                self.task_hash[task_id].append(new_item)

        del self.hash[key]
        self._deleted[key] = True

    def add(self, stat, cell_id, increment=1):
        episode_step = stat['episode_step']
        task_id = stat['task_id']

        # episode_step should be greater than 0.
        if episode_step > 0:
            _hash = self._prepare_to_add_cell(stat)
            self._prepare_to_add_task(task_id)

            if not cell_id in _hash:
                _hash[cell_id] = 0

            _hash[cell_id] += increment
            _hash['steps'] += 1
            _hash['score'] += stat['score']

            assert episode_step == _hash['steps']

            cnt = _hash[cell_id]
            out = CellCount(cnt)
            out.steps = _hash['steps']

            if self.with_best_episode:
                episodes = self.task_hash[task_id]
                if len(episodes) > 0:
                    out.best_epi_cnt = episodes.count(cell_id)
        else:
            cnt = 1
            out = CellCount(cnt)

        if stat['done']:
            self.finalize(stat)

        return out


class VQCellRep(nn.Module):
    ARCH = {
        'vq': 'basic',
        'vqrl': 'resl',
    }
    def __init__(self, cfg,
                 obs_shape=DMLAB_OBS_SHAPE,
                 ext_dim=None):
        super().__init__()
        self.cfg = cfg
        self.cell_spec = CellSpec(cfg.cell_spec, type=cfg.cell_type)

        print(self.cell_spec)

        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')
        if self.is_minigrid:
            self.max_obs_val = torch.FloatTensor([10., 5., 2.]).expand(1, 1, 1, 3).permute(0, 3, 1, 2)

        my_obs_shape = (obs_shape[0] - self.cell_spec.offset[0], obs_shape[1] - self.cell_spec.offset[1])
        _transforms = []
        if obs_shape != my_obs_shape:
            _transforms.append(transforms.CenterCrop(my_obs_shape))
        if self.is_dmlab:
            target_shape = self.cell_spec.resize_target_shape
        elif self.is_minigrid:
            proportion = 2 ** 2
            target_shape = (self.cell_spec.resolution[0] * proportion, self.cell_spec.resolution[1] * proportion)
        _transforms.append(transforms.Resize(target_shape))
        self.transforms = transforms.Compose(_transforms)

        self.reg_type, self.reg_coef = cfg.cell_reg.split(',')
        self.reg_coef = float(self.reg_coef)

        self._build(ext_dim=ext_dim)

    def _build(self, ext_dim=None):
        arch = self.ARCH[self.cell_spec.vq_type]
        num_blocks = self.cell_spec.num_res_blocks
        if self.is_minigrid:
            if ext_dim is None:
                from .vqvae import VectorQuantizedVAEforMG
                self.model = VectorQuantizedVAEforMG(3, self.cfg.cell_dim, K=self.cell_spec.depth, num_blocks=num_blocks,
                                                reg_type=self.reg_type, arch=arch)
                self.extended = False
            else:
                from .vqvae import ExtendedVectorQuantizedVAEforMG
                self.model = ExtendedVectorQuantizedVAEforMG(3, self.cfg.cell_dim, K=self.cell_spec.depth, ext_dim=ext_dim, num_blocks=num_blocks, reg_type=self.reg_type, arch=arch)
                self.extended = True
        else:
            if ext_dim is None:
                from .vqvae import VectorQuantizedVAE
                self.model = VectorQuantizedVAE(3, self.cfg.cell_dim, K=self.cell_spec.depth, num_cnn_blocks=self.cell_spec.num_cnn_blocks, num_res_blocks=self.cell_spec.num_res_blocks, reg_type=self.reg_type, arch=arch)
                self.extended = False
            else:
                from .vqvae import ExtendedVectorQuantizedVAE
                self.model = ExtendedVectorQuantizedVAE(3, self.cfg.cell_dim, K=self.cell_spec.depth, ext_dim=ext_dim, num_blocks=num_blocks, reg_type=self.reg_type, arch=arch)
                self.extended = True

    def get_obs_norm(self, obs):
        if self.is_minigrid:
            return (obs / self.max_obs_val.to(obs.device))*2.0 - 1.0
        else:
            return (obs - self.cfg.obs_subtract_mean) / self.cfg.obs_scale

    def calc_loss(self, obs, output_dict, ext_data=None):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)

        if self.extended:
            assert ext_data is not None
            loss_recons, loss_vq, loss_commit, loss_reg = self.model.calc_losses(obs, ext_data)
        else:
            loss_recons, loss_vq, loss_commit, loss_reg = self.model.calc_losses(obs)
        loss = loss_recons + loss_vq + loss_commit
        if self.reg_coef > 0:
            loss += self.reg_coef * loss_reg

        output_dict['loss_recons'] = loss_recons
        output_dict['loss_vq'] = loss_vq
        output_dict['loss_reg'] = loss_reg

        return loss

    def get_representations(self, ids):
        return self.model.select_codes(ids)

    def forward(self, obs, ext_data=None, return_reps=False):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)
        if self.extended:
            return self.model.encode(obs, ext_data, return_reps=return_reps)
        else:
            return self.model.encode(obs, return_reps=return_reps)


class AECellRep(VQCellRep):
    ARCH = {
        'ae': 'basic',
        'aerl': 'resl',
    }
    def __init__(self, cfg,
                 obs_shape=DMLAB_OBS_SHAPE,
                 ext_dim=None, D=512):


        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')

        self.D = cfg.cell_enc_hidden
        super(AECellRep, self).__init__(cfg, obs_shape=obs_shape, ext_dim=ext_dim)

        self.K = cfg.hash_dim
        self.A = nn.parameter.Parameter(data=torch.normal(0, 1, size=(self.D, self.K)))
        self.coef = self.reg_coef

    def _build(self, ext_dim=None):
        arch = self.ARCH[self.cell_spec.vq_type]
        num_blocks = self.cell_spec.num_res_blocks
        self.extended = False

        if self.is_minigrid:
            from .vqvae import AE4SimHashForMG
            self.model = AE4SimHashForMG(3, self.cfg.cell_dim, D=self.D, num_blocks=num_blocks, arch=arch, resolution=self.cell_spec.resolution)
        else:
            from .vqvae import AE4SimHash
            self.model = AE4SimHash(3, self.cfg.cell_dim, D=self.D, num_blocks=num_blocks, arch=arch, resolution=self.cell_spec.resolution)

    def calc_loss(self, obs, output_dict, ext_data=None):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)

        loss_recons, loss_binarization = self.model.calc_losses(obs)

        loss = loss_recons + self.coef * loss_binarization

        output_dict['loss_recons'] = loss_recons
        output_dict['loss_vq'] = loss_binarization

        return loss

    def forward(self, obs, ext_data=None, return_reps=False):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)

        if return_reps:
            codes, reps = self.model.encode(obs, return_reps=return_reps)
        else:
            codes = self.model.encode(obs)

        with torch.no_grad():
            cells = torch.matmul(codes, self.A)
            cells = torch.sign(cells).clip(0, 1)

        if return_reps:
            return cells, reps
        return cells


class DSCellRep(nn.Module):
    def __init__(self, cfg,
                 obs_shape=DMLAB_OBS_SHAPE,
                 colored=False):
        super().__init__()

        self.cfg = cfg
        self.colored = colored
        self.cell_spec = CellSpec(cfg.cell_spec, type=cfg.cell_type, is_ds_cell=True)
        self.obs_shape = obs_shape

        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')
        if self.is_minigrid:
            self.max_obs_val = torch.FloatTensor([10., 5., 2.]).expand(1,1,1,3).permute(0, 3, 1, 2)

        print(self.cell_spec)

        if 'random' in cfg.cell_type:
            self.random_update()
        else:
            self.update()

    def random_update(self):
        # this method is defined to seek non-stationary issue of count-based intrinsic rewards
        candidates = ['30,0,1x4x4', '30,10,1x7x3', '30,0,1x3x10', '30,30,2x2x6', '30,20,2x4x4']
        cell_spec = CellSpec(random.choice(candidates), is_ds_cell=True)
        while self.cell_spec == cell_spec:
            cell_spec = CellSpec(random.choice(candidates), is_ds_cell=True)
        prev = self.cell_spec
        if cell_spec.offset is None:
            cell_spec.offset = prev.offset
        self.cell_spec = cell_spec
        self.update()
        return f'prev:{prev}, updated:{cell_spec}'

    def update(self, cell_spec=None):
        cell_spec = cell_spec or self.cell_spec
        my_obs_shape = (self.obs_shape[0] - cell_spec.offset[0], self.obs_shape[1] - cell_spec.offset[1])
        _transforms = []
        if self.obs_shape != my_obs_shape:
            _transforms.append(transforms.CenterCrop(my_obs_shape))
        _transforms.append(transforms.Resize(cell_spec.resolution, interpolation=transforms.InterpolationMode.NEAREST))
        if not self.colored:
            _transforms.append(transforms.Grayscale())
        self.transforms = transforms.Compose(_transforms)

    def forward(self, obs, **kwargs):
        obs = self.transforms(obs)  # n x w x h
        assert obs.dim() == 4
        if self.is_dmlab:
            obs /= 255.0
        elif self.is_minigrid:
            obs /= self.max_obs_val.to(obs.device)
        if obs.dim() == 4:
            obs = obs.squeeze(dim=1)
        cells = (obs * self.cell_spec.depth)
        return cells

    def get_representations(self, ids):
        return ids


class CellCounter(nn.Module):
    def __init__(self, cfg, action_space=None):
        super().__init__()

        self.cfg = cfg
        self.cell_storage = EpisodicCells()

        is_dmlab = cfg.env.lower().startswith('dmlab')
        is_minigrid = cfg.env.lower().startswith('minigrid')
        self.cell_rep = None
        self.use_random_update = False
        cell_type = self.cfg.cell_type
        if cell_type is not None and (is_dmlab or is_minigrid) and cell_type != 'none':
            if is_dmlab:
                obs_shape = DMLAB_OBS_SHAPE
            elif is_minigrid:
                obs_shape = MINIGRID_OBS_SHAPE

            self.use_random_update = 'random' in cell_type
            cell_rep_class = None
            kwargs = {}
            if cell_type.startswith('ds'):
                cell_rep_class = DSCellRep
                if cell_type.startswith('dsc'):
                    kwargs['colored'] = True
            elif cell_type.startswith('vq'):
                cell_rep_class = VQCellRep
                if 'ext' in cell_type:
                    assert action_space is not None
                    # kwargs['ext_dim'] = action_space.n + 1
                    kwargs['ext_dim'] = action_space.n

                if 'instr' in cell_type:
                    if 'ext_dim' in kwargs:
                        kwargs['ext_dim'] += 64
                    else:
                        kwargs['ext_dim'] = 64
            elif cell_type.startswith('ae'):
                cell_rep_class = AECellRep

            if cell_type.endswith('list'):
                self.cell_rep = [cell_rep_class(cfg, obs_shape, **kwargs) for _ in range(cfg.num_envs)]
            else:
                self.cell_rep = cell_rep_class(cfg, obs_shape, **kwargs)

        if cell_type is not None:
            self.learnable_hash = self.cfg.cell_type.startswith('vq') or self.cfg.cell_type.startswith('ae')
        else:
            self.learnable_hash = False
        self._has_single_cell_rep = not type(self.cell_rep) == list
        self._freezing = self.learnable_hash and self.cfg.freeze_vq
        self._maybe_load_pretrained_cell_rep()

    def _maybe_load_pretrained_cell_rep(self):
        if self.learnable_hash and self.cfg.pretrained_vq_path is not None:
            ckpt = torch.load(self.cfg.pretrained_vq_path)
            if self._has_single_cell_rep:
                self.cell_rep.load_state_dict(ckpt, strict=False)
            else:
                for _cell_rep in self.cell_rep:
                    _cell_rep.load_state_dict(ckpt, strict=False)
            print(f'Successfully loaded pretrained vq-rep from {self.cfg.pretrained_vq_path}')

    def calc_loss(self, obs, output_dict, ext_data=None):
        if self._freezing:
            with torch.no_grad():
                return self.cell_rep.calc_loss(obs, output_dict, ext_data=ext_data)
        else:
            return self.cell_rep.calc_loss(obs, output_dict, ext_data=ext_data)

    @property
    def cell_dim(self):
        cell_type = self.cfg.cell_type.split('_')[0]
        if cell_type == 'ds':
            return 1
        elif cell_type == 'dsc':
            return 3
        elif cell_type.startswith('vq'):
            return self.cfg.cell_dim
        elif cell_type.startswith('ae'):
            return self.cfg.cell_dim

    def obs2cell(self, obs, to_numpy=True, stats=None, ext_data=None, return_reps=False):
        reps = None
        if self.cell_rep is not None:
            if return_reps:
                cells, reps = self.cell_rep(obs, ext_data=ext_data, return_reps=return_reps)
            else:
                cells = self.cell_rep(obs, ext_data=ext_data)
        else:
            cells = obs

        if to_numpy and type(cells) == torch.Tensor:
            cells = cells.detach().cpu().numpy().astype(np.uint8)

        if return_reps:
            return cells, reps
        return cells

    def get_representations(self, ids):
        return self.cell_rep.get_representations(ids)

    @staticmethod
    def cell2key(cell, act=None, rew=None):
        items = cell.reshape(-1).tolist()
        if act is not None:
            items.append(act)
        if rew is not None:
            items.append(rew)
        key = '_'.join(map(str, items))
        return key

    def _quantize_reward(self, rewards):
        result = torch.where(rewards > 0, torch.ones_like(rewards), rewards)
        result = torch.where(rewards < 0, -torch.ones_like(rewards), result)
        return result.int()

    def prepare_items(self, obs, stats, ext_data=None, actions=None, rewards=None):
        cells = self.obs2cell(obs, stats=stats, ext_data=ext_data)
        if actions is not None:
            actions = actions.detach().cpu().numpy().astype(np.uint8)
        return cells, actions, rewards

    def count(self, stats, cells, actions=None, rewards=None):
        rets = []
        for i, (cell, stat) in enumerate(zip(cells, stats)):
            act = None if actions is None else actions[i]
            rew = None if rewards is None else rewards[i]
            cell_key = self.cell2key(cell, act=act, rew=rew)
            out = self.cell_storage.add(stat, cell_key)
            rets.append(out)

        return rets

    def add(self, obs, stats, ext_data=None, actions=None, rewards=None):
        cells, actions, rewards = self.prepare_items(obs, stats, ext_data=ext_data, actions=actions, rewards=rewards)
        outs = self.count(stats, cells, actions, rewards)
        return CellCountList(outs, cells)