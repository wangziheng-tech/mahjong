from bisect import bisect_right
from itertools import permutations

import numpy as np
from torch.utils.data import Dataset

OBS_CHANNELS = 18
OBS_TILES = 36
ACT_SIZE = 235

SUIT_PERMS = np.array(list(permutations(range(3))), dtype=np.int64)


def _build_tile_perms():
    tile_perms = []
    for suit_perm in SUIT_PERMS:
        tile_perm = np.arange(OBS_TILES, dtype=np.int64)
        for old_suit, new_suit in enumerate(suit_perm):
            old_start = old_suit * 9
            new_start = int(new_suit) * 9
            tile_perm[old_start:old_start + 9] = np.arange(new_start, new_start + 9)
        tile_perms.append(tile_perm)
    return np.stack(tile_perms, axis=0)


def _build_action_perms(tile_perms):
    action_perms = []
    for suit_perm, tile_perm in zip(SUIT_PERMS, tile_perms):
        action_perm = np.arange(ACT_SIZE, dtype=np.int64)
        tile_actions = tile_perm[:34]

        for start in (2, 99, 133, 167, 201):
            action_perm[start:start + 34] = start + tile_actions

        for old_offset in range(63):
            old_suit = old_offset // 21
            rest = old_offset % 21
            new_suit = int(suit_perm[old_suit])
            action_perm[36 + old_offset] = 36 + new_suit * 21 + rest

        action_perms.append(action_perm)
    return np.stack(action_perms, axis=0)


TILE_PERMS = _build_tile_perms()
ACTION_PERMS = _build_action_perms(TILE_PERMS)


class MahjongGBDataset(Dataset):

    def __init__(self, begin=0, end=1, augment=False):
        import json
        with open('data_v2/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin:self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0:
                print('loading', i)
            d = np.load('data_v2/%d.npz' % (i + self.begin))
            for k in d:
                self.cache[k].append(d[k])

    def __len__(self):
        return self.samples

    def _augment_sample(self, obs, mask, act):
        perm_id = np.random.randint(len(SUIT_PERMS))
        if perm_id == 0:
            return obs, mask, int(act)

        tile_perm = TILE_PERMS[perm_id]
        action_perm = ACTION_PERMS[perm_id]

        aug_obs = np.empty_like(obs)
        obs_flat = obs.reshape(OBS_CHANNELS, OBS_TILES)
        aug_flat = aug_obs.reshape(OBS_CHANNELS, OBS_TILES)
        aug_flat[:, tile_perm] = obs_flat

        aug_mask = np.zeros_like(mask)
        aug_mask[action_perm] = mask

        return aug_obs, aug_mask, int(action_perm[int(act)])

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        obs = self.cache['obs'][match_id][sample_id]
        mask = self.cache['mask'][match_id][sample_id]
        act = self.cache['act'][match_id][sample_id]
        if self.augment:
            obs, mask, act = self._augment_sample(obs, mask, act)
        return obs, mask, act
