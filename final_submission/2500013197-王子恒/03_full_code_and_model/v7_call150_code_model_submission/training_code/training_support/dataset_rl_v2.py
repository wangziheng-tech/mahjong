from bisect import bisect_right

import numpy as np

from dataset_v2 import MahjongGBDataset


class MahjongGBRLDataset(MahjongGBDataset):
    def __init__(self, begin=0, end=1, augment=False, reward_dir='data_rl_v2', reward_scale=64.0):
        super().__init__(begin, end, augment)
        self.reward_dir = reward_dir
        self.reward_scale = reward_scale
        self.cache['reward'] = []
        self.cache['player'] = []
        for i in range(self.matches):
            d = np.load('%s/%d.npz' % (reward_dir, i + self.begin))
            if len(d['reward']) != len(self.cache['act'][i]):
                raise RuntimeError('reward length mismatch in match %d' % (i + self.begin))
            self.cache['reward'].append(d['reward'])
            self.cache['player'].append(d['player'])

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        obs = self.cache['obs'][match_id][sample_id]
        mask = self.cache['mask'][match_id][sample_id]
        act = self.cache['act'][match_id][sample_id]
        reward = self.cache['reward'][match_id][sample_id] / self.reward_scale
        player = self.cache['player'][match_id][sample_id]
        if self.augment:
            obs, mask, act = self._augment_sample(obs, mask, act)
        return obs, mask, act, np.float32(reward), np.int64(player)
